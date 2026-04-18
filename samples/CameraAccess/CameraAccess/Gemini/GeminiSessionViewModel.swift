import Foundation
import SwiftUI
import Combine

@MainActor
class GeminiSessionViewModel: ObservableObject {
  @Published var isGeminiActive: Bool = false
  @Published var connectionState: GeminiConnectionState = .disconnected
  @Published var isModelSpeaking: Bool = false
  @Published var errorMessage: String?
  @Published var openClawErrorMessage: String?
  @Published var userTranscript: String = ""
  @Published var aiTranscript: String = ""
  @Published var isListeningForSpeech: Bool = false
  @Published var toolCallStatus: ToolCallStatus = .idle
  @Published var openClawConnectionState: OpenClawConnectionState = .notConfigured
  private let geminiService = GeminiLiveService()
  private let openClawBridge = OpenClawBridge()
  private var toolCallRouter: ToolCallRouter?
  private let audioManager = AudioManager()
  private let eventClient = OpenClawEventClient()
  private let speechManager = SpeechToTextManager()
  private var cancellables = Set<AnyCancellable>()
  private var lastVideoFrameTime: Date = .distantPast
  private var stateObservation: Task<Void, Never>?
  private var reconnectTask: Task<Void, Never>?

  /// Set by parent view to trigger photo capture on the stream session
  var onCapturePhoto: (() -> Void)?

  var streamingMode: StreamingMode = .glasses

  func startSession() async {
    guard !isGeminiActive else { return }

    guard GeminiConfig.isConfigured else {
      errorMessage = "Gemini API key not configured. Open GeminiConfig.swift and replace YOUR_GEMINI_API_KEY with your key from https://aistudio.google.com/apikey"
      return
    }

    isGeminiActive = true

    // Hook up SFSpeech text transcriptions to Gemini via sendText
    speechManager.$finalTranscribedText
      .receive(on: DispatchQueue.main)
      .sink { [weak self] text in
        guard let self = self, !text.isEmpty else { return }
        self.sendTextMessage(text)
      }
      .store(in: &cancellables)

    speechManager.$partialTranscribedText
      .receive(on: DispatchQueue.main)
      .sink { [weak self] partial in
        guard let self = self, !partial.isEmpty else { return }
        self.userTranscript = partial
      }
      .store(in: &cancellables)

    speechManager.$isListening
      .receive(on: DispatchQueue.main)
      .assign(to: &$isListeningForSpeech)

    // Wire audio callbacks (we bypass Gemini's audio pipeline completely now
    // to strictly rely on our SFSpeechRecognizer text)
    audioManager.onAudioCaptured = { [weak self] data in
      // Optionally we could send audio AND text, but doing text-only 
      // completely prevents Gemini hallucinations from its native decoder.
    }

    audioManager.onAudioBuffer = { [weak self] buffer in
      self?.speechManager.processBuffer(buffer)
    }

    geminiService.onAudioReceived = { [weak self] data in
      self?.audioManager.playAudio(data: data)
    }

    geminiService.onInterrupted = { [weak self] in
      self?.audioManager.stopPlayback()
    }

    geminiService.onTurnComplete = { [weak self] in
      guard let self else { return }
      Task { @MainActor in
        // Clear user transcript when AI finishes responding
        self.userTranscript = ""
      }
    }

    geminiService.onInputTranscription = { [weak self] text in
      guard let self else { return }
      Task { @MainActor in
        self.userTranscript += text
      }
    }

    geminiService.onOutputTranscription = { [weak self] text in
      guard let self else { return }
      Task { @MainActor in
        self.aiTranscript += text
        self.userTranscript = ""
      }
    }

    // Handle unexpected disconnection
    geminiService.onDisconnected = { [weak self] reason in
      guard let self else { return }
      Task { @MainActor in
        guard self.isGeminiActive else { return }
        self.connectionState = .disconnected
        self.geminiService.disconnect()
        self.errorMessage = "Connection lost: \(reason ?? "Unknown error"). Reconnecting..."
        self.scheduleReconnection()
      }
    }

    // Check OpenClaw connectivity and start fresh session
    await openClawBridge.checkConnection()
    openClawBridge.resetSession()

    // Wire tool call handling
    toolCallRouter = ToolCallRouter(bridge: openClawBridge)
    toolCallRouter?.onCapturePhoto = { [weak self] in
      self?.onCapturePhoto?()
    }

    geminiService.onToolCall = { [weak self] toolCall in
      guard let self else { return }
      Task { @MainActor in
        for call in toolCall.functionCalls {
          self.toolCallRouter?.handleToolCall(call) { [weak self] response in
            self?.geminiService.sendToolResponse(response)
          }
        }
      }
    }

    geminiService.onToolCallCancellation = { [weak self] cancellation in
      guard let self else { return }
      Task { @MainActor in
        self.toolCallRouter?.cancelToolCalls(ids: cancellation.ids)
      }
    }

    // Observe service state
    stateObservation = Task { [weak self] in
      guard let self else { return }
      while !Task.isCancelled {
        try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
        guard !Task.isCancelled else { break }
        self.connectionState = self.geminiService.connectionState
        self.isModelSpeaking = self.geminiService.isModelSpeaking
        self.toolCallStatus = self.openClawBridge.lastToolCallStatus
        self.openClawConnectionState = self.openClawBridge.connectionState
        if case .unreachable(let msg) = self.openClawConnectionState {
            if self.openClawErrorMessage == nil {
                self.openClawErrorMessage = "OpenClaw Gateway Error: \(msg)"
            }
            // Attempt to ping OpenClaw occasionally to recover automatically
            if Int.random(in: 0..<50) == 0 { 
                Task { [weak self] in await self?.openClawBridge.checkConnection() }
            }
        } else {
            self.openClawErrorMessage = nil       
        }
      }
    }

    // Setup audio
    do {
      try audioManager.setupAudioSession(useIPhoneMode: streamingMode == .iPhone)
    } catch {
      errorMessage = "Audio setup failed: \(error.localizedDescription)"
      isGeminiActive = false
      return
    }

    // Connect to Gemini and wait for setupComplete
    let setupOk = await geminiService.connect()

    if !setupOk {
      let msg: String
      if case .error(let err) = geminiService.connectionState {
        msg = err
      } else {
        msg = "Failed to connect to Gemini"
      }
      errorMessage = msg
      geminiService.disconnect()
      stateObservation?.cancel()
      stateObservation = nil
      isGeminiActive = false
      connectionState = .disconnected
      return
    }

    // Start mic capture AND start Apple Speech recognizer
    do {
      try audioManager.startCapture()
      
      // Request auth if needed and start the powerful local speech deciphering
      speechManager.requestAuthorization { [weak self] authorized in
        guard let self = self, authorized else { return }
        do {
          try self.speechManager.startProcessing(from: self.audioManager.audioEngine)
        } catch {
          print("Local speech fallback failed: \(error.localizedDescription)")
        }
      }
    } catch {
      errorMessage = "Mic capture failed: \(error.localizedDescription)"
      geminiService.disconnect()
      stateObservation?.cancel()
      stateObservation = nil
      isGeminiActive = false
      connectionState = .disconnected
      return
    }

    // Connect to OpenClaw event stream for proactive notifications
    if SettingsManager.shared.proactiveNotificationsEnabled {
      eventClient.onNotification = { [weak self] text in
        guard let self else { return }
        Task { @MainActor in
          guard self.isGeminiActive, self.connectionState == .ready else { return }
          self.geminiService.sendTextMessage(text)
        }
      }
      eventClient.connect()
    }
  }

  func stopSession() {
    eventClient.disconnect()
    reconnectTask?.cancel()
    reconnectTask = nil
    toolCallRouter?.cancelAll()
    toolCallRouter = nil
    audioManager.stopCapture()
    speechManager.cancelProcessing()
    geminiService.disconnect()
    stateObservation?.cancel()
    stateObservation = nil
    isGeminiActive = false
    connectionState = .disconnected
    isModelSpeaking = false
    userTranscript = ""
    aiTranscript = ""
    toolCallStatus = .idle
  }

  func sendVideoFrameIfThrottled(image: UIImage) {
    guard SettingsManager.shared.videoStreamingEnabled else { return }
    guard isGeminiActive, connectionState == .ready else { return }
    let now = Date()
    guard now.timeIntervalSince(lastVideoFrameTime) >= GeminiConfig.videoFrameInterval else { return }
    lastVideoFrameTime = now
    geminiService.sendVideoFrame(image: image)
  }

  /// Send a typed text message to Gemini, bypassing the audio pipeline.
  func sendTextMessage(_ text: String) {
    guard isGeminiActive, connectionState == .ready, !text.isEmpty else { return }
    userTranscript = text
    aiTranscript = ""
    geminiService.sendText(text)
  }

  func restartSpeechDictation() {
    guard isGeminiActive else { return }
    do {
      try speechManager.startProcessing(from: audioManager.audioEngine)
    } catch {
      print("Failed to manually restart dictation: \(error.localizedDescription)")
    }
  }

  private func scheduleReconnection() {
    reconnectTask?.cancel()
    reconnectTask = Task { [weak self] in
      try? await Task.sleep(nanoseconds: 3_000_000_000) // 3 seconds
      guard let self = self, !Task.isCancelled, self.isGeminiActive else { return }
      self.errorMessage = "Attempting to reconnect..."
      
      // Update OpenClaw status
      await self.openClawBridge.checkConnection()
      
      // Attempt connection to Gemini again
      let setupOk = await self.geminiService.connect()
      if setupOk {
        self.errorMessage = nil
        do {
          try self.audioManager.startCapture()
          self.restartSpeechDictation()
        } catch {
          self.errorMessage = "Mic error on reconnect: \(error.localizedDescription)"
        }
      } else {
        self.scheduleReconnection()
      }
    }
  }

}
