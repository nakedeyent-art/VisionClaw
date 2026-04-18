import AVFoundation
import Speech
import Foundation
import UIKit

class AudioManager {
  var onAudioCaptured: ((Data) -> Void)?
  var onAudioBuffer: ((AVAudioPCMBuffer) -> Void)?

  let audioEngine = AVAudioEngine()
  private let playerNode = AVAudioPlayerNode()
  private var isCapturing = false
  private var wasCapturingBeforeInterruption = false
  private var useIPhoneMode = false

  private let outputFormat: AVAudioFormat

  // Accumulate resampled PCM into ~100ms chunks before sending
  private let sendQueue = DispatchQueue(label: "audio.accumulator")
  private var accumulatedData = Data()
  private let minSendBytes = 3200  // 100ms at 16kHz mono Int16 = 1600 frames * 2 bytes

  // Notification observers for background resilience
  private var interruptionObserver: NSObjectProtocol?
  private var routeChangeObserver: NSObjectProtocol?
  private var mediaServicesResetObserver: NSObjectProtocol?
  private var foregroundObserver: NSObjectProtocol?

  init() {
    self.outputFormat = AVAudioFormat(
      commonFormat: .pcmFormatInt16,
      sampleRate: GeminiConfig.outputAudioSampleRate,
      channels: GeminiConfig.audioChannels,
      interleaved: true
    )!
  }

  func setupAudioSession(useIPhoneMode: Bool = false) throws {
    self.useIPhoneMode = useIPhoneMode
    let session = AVAudioSession.sharedInstance()
    let forceSpeaker = SettingsManager.shared.speakerOutputEnabled

    if useIPhoneMode || forceSpeaker {
      // iPhone mode: use built-in mic + speaker with aggressive echo cancellation
      try session.setCategory(
        .playAndRecord,
        mode: .voiceChat,
        options: [.defaultToSpeaker, .allowBluetooth, .allowBluetoothA2DP, .mixWithOthers]
      )
      // Force built-in mic for iPhone mode
      if let availableInputs = session.availableInputs,
         let builtInMic = availableInputs.first(where: { $0.portType == .builtInMic }) {
        try session.setPreferredInput(builtInMic)
        NSLog("[Audio] ✅ iPhone mode: forced built-in mic")
      }
    } else {
      // Glasses mode: use the glasses' Bluetooth mic (right next to user's mouth!)
      // This is WHY the Meta app works perfectly — it uses this same mic.
      // .allowBluetoothHFP routes audio INPUT through the glasses' mic.
      // .videoChat uses mild AEC appropriate for glasses (mic + speaker both on glasses).
      try session.setCategory(
        .playAndRecord,
        mode: .videoChat,
        options: [.allowBluetoothHFP, .allowBluetoothA2DP, .defaultToSpeaker, .mixWithOthers]
      )
      // Let iOS pick the Bluetooth mic (glasses) — do NOT force built-in mic
      // The manual downsample handles any sample rate the BT device provides
      if let availableInputs = session.availableInputs {
        NSLog("[Audio] Available inputs: %@", availableInputs.map { "\($0.portName) (\($0.portType.rawValue))" }.joined(separator: ", "))
        // Prefer Bluetooth input (glasses mic) if available
        if let btMic = availableInputs.first(where: {
          $0.portType == .bluetoothHFP || $0.portType == .bluetoothA2DP || $0.portType == .bluetoothLE
        }) {
          try session.setPreferredInput(btMic)
          NSLog("[Audio] ✅ Glasses mode: preferred Bluetooth mic: %@", btMic.portName)
        } else {
          NSLog("[Audio] ⚠️ No Bluetooth mic found — falling back to default input")
        }
      }
    }

    try session.setPreferredSampleRate(GeminiConfig.inputAudioSampleRate)
    try session.setPreferredIOBufferDuration(0.064)
    try session.setActive(true)
    // Verify hardware actually honored our preferred sample rate
    let actualRate = session.sampleRate
    if actualRate != GeminiConfig.inputAudioSampleRate {
      NSLog("[Audio] ⚠️ Hardware rate=%.0fHz (requested %.0fHz) — resampling REQUIRED",
            actualRate, GeminiConfig.inputAudioSampleRate)
    } else {
      NSLog("[Audio] ✅ Hardware rate=%.0fHz matches target", actualRate)
    }

    // Log the actual audio route for diagnostics
    let currentRoute = session.currentRoute
    for input in currentRoute.inputs {
      NSLog("[Audio] 🎤 Active input: %@ (%@) channels=%d",
            input.portName, input.portType.rawValue, input.channels?.count ?? 0)
    }
    for output in currentRoute.outputs {
      NSLog("[Audio] 🔊 Active output: %@ (%@)", output.portName, output.portType.rawValue)
    }

    if SettingsManager.shared.speakerOutputEnabled {
      try session.overrideOutputAudioPort(.speaker)
      NSLog("[Audio] Speaker output override: ON (iPhone speaker)")
    }
    NSLog("[Audio] Session mode: %@", useIPhoneMode ? "voiceChat (iPhone mic)" : "videoChat (glasses BT mic)")

    setupInterruptionHandling()
    setupAppLifecycleObservers()
  }

  func startCapture() throws {
    guard !isCapturing else { return }

    audioEngine.attach(playerNode)
    let playerFormat = AVAudioFormat(
      commonFormat: .pcmFormatFloat32,
      sampleRate: GeminiConfig.outputAudioSampleRate,
      channels: GeminiConfig.audioChannels,
      interleaved: false
    )!
    audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: playerFormat)

    let inputNode = audioEngine.inputNode
    let inputNativeFormat = inputNode.outputFormat(forBus: 0)

    NSLog("[Audio] Native input format: %@ sampleRate=%.0f channels=%d",
          inputNativeFormat.commonFormat == .pcmFormatFloat32 ? "Float32" :
          inputNativeFormat.commonFormat == .pcmFormatInt16 ? "Int16" : "Other",
          inputNativeFormat.sampleRate, inputNativeFormat.channelCount)

    let nativeRate = inputNativeFormat.sampleRate
    let nativeChannels = Int(inputNativeFormat.channelCount)
    let targetRate = GeminiConfig.inputAudioSampleRate  // 16000

    NSLog("[Audio] Manual downsample pipeline: %.0fHz %dch → %.0fHz 1ch",
          nativeRate, nativeChannels, targetRate)

    sendQueue.async { self.accumulatedData = Data() }

    var tapCount = 0
    // Tap in NATIVE format (always works on all hardware), then manually convert
    inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputNativeFormat) { [weak self] buffer, _ in
      guard let self else { return }
      self.onAudioBuffer?(buffer)

      tapCount += 1
      let pcmData = self.manualDownsampleToInt16(
        buffer, inputRate: nativeRate, outputRate: targetRate, inputChannels: nativeChannels)

      if tapCount <= 5 {
        NSLog("[Audio] Tap #%d: %d input frames → %d output bytes (%.0fHz %dch → %.0fHz 1ch)",
              tapCount, buffer.frameLength, pcmData.count,
              nativeRate, nativeChannels, targetRate)
      }

      guard !pcmData.isEmpty else { return }

      // Accumulate into ~100ms chunks before sending to Gemini
      self.sendQueue.async {
        self.accumulatedData.append(pcmData)
        if self.accumulatedData.count >= self.minSendBytes {
          let chunk = self.accumulatedData
          self.accumulatedData = Data()
          if tapCount <= 5 {
            NSLog("[Audio] Sending chunk: %d bytes (~%dms)",
                  chunk.count, chunk.count / 32)
          }
          self.onAudioCaptured?(chunk)
        }
      }
    }

    try audioEngine.start()
    playerNode.play()
    isCapturing = true
  }

  /// Manual downsample: native Float32 (any rate, any channels) → 16kHz mono Int16 PCM.
  /// Uses linear interpolation for sample rate conversion and channel averaging for stereo→mono.
  /// Zero dependence on Apple's AVAudioConverter or MixerNode (both failed on iPhone 16 Pro Max).
  private func manualDownsampleToInt16(
    _ buffer: AVAudioPCMBuffer,
    inputRate: Double,
    outputRate: Double,
    inputChannels: Int
  ) -> Data {
    let inputFrames = Int(buffer.frameLength)
    guard inputFrames > 0 else { return Data() }

    // Step 1: Get mono float samples (average channels if stereo)
    var monoSamples = [Float](repeating: 0, count: inputFrames)
    if let floatData = buffer.floatChannelData {
      if inputChannels == 1 {
        // Mono: direct copy
        for i in 0..<inputFrames {
          monoSamples[i] = floatData[0][i]
        }
      } else {
        // Stereo+: average all channels
        for i in 0..<inputFrames {
          var sum: Float = 0
          for ch in 0..<inputChannels {
            sum += floatData[ch][i]
          }
          monoSamples[i] = sum / Float(inputChannels)
        }
      }
    } else {
      return Data()  // Not Float32 format
    }

    // Step 2: Resample with anti-aliasing (windowed average, not linear interpolation).
    // Linear interpolation causes high-frequency folding artifacts that confuse speech recognition.
    // Windowed averaging acts as a natural low-pass filter: for each output sample,
    // average ALL input samples that fall within its window [i*ratio, (i+1)*ratio).
    let outputSamples: [Float]
    if abs(inputRate - outputRate) < 1.0 {
      // Same rate — no resampling needed
      outputSamples = monoSamples
    } else {
      let ratio = inputRate / outputRate  // e.g., 48000/16000 = 3.0
      let outputFrames = Int(Double(inputFrames) / ratio)
      guard outputFrames > 0 else { return Data() }
      var resampled = [Float](repeating: 0, count: outputFrames)
      for i in 0..<outputFrames {
        // Window of input samples for this output sample
        let windowStart = Int(Double(i) * ratio)
        let windowEnd = min(Int(Double(i + 1) * ratio), inputFrames)
        let windowSize = max(windowEnd - windowStart, 1)
        var sum: Float = 0
        for j in windowStart..<windowEnd {
          sum += monoSamples[j]
        }
        resampled[i] = sum / Float(windowSize)
      }
      outputSamples = resampled
    }

    // Step 3: Convert Float32 [-1.0, 1.0] → Int16 [-32767, 32767]
    var int16Array = [Int16](repeating: 0, count: outputSamples.count)
    for i in 0..<outputSamples.count {
      let clamped = max(-1.0, min(1.0, outputSamples[i]))
      int16Array[i] = Int16(clamped * Float(Int16.max))
    }

    return Data(bytes: &int16Array, count: int16Array.count * MemoryLayout<Int16>.size)
  }

  func playAudio(data: Data) {
    guard isCapturing, !data.isEmpty else { return }

    let playerFormat = AVAudioFormat(
      commonFormat: .pcmFormatFloat32,
      sampleRate: GeminiConfig.outputAudioSampleRate,
      channels: GeminiConfig.audioChannels,
      interleaved: false
    )!

    let frameCount = UInt32(data.count) / (GeminiConfig.audioBitsPerSample / 8 * GeminiConfig.audioChannels)
    guard frameCount > 0 else { return }

    guard let buffer = AVAudioPCMBuffer(pcmFormat: playerFormat, frameCapacity: frameCount) else { return }
    buffer.frameLength = frameCount

    guard let floatData = buffer.floatChannelData else { return }
    data.withUnsafeBytes { rawBuffer in
      guard let int16Ptr = rawBuffer.bindMemory(to: Int16.self).baseAddress else { return }
      for i in 0..<Int(frameCount) {
        floatData[0][i] = Float(int16Ptr[i]) / Float(Int16.max)
      }
    }

    playerNode.scheduleBuffer(buffer)
    if !playerNode.isPlaying {
      playerNode.play()
    }
  }

  func stopPlayback() {
    playerNode.stop()
    playerNode.play()
  }

  func stopCapture() {
    guard isCapturing else { return }
    audioEngine.inputNode.removeTap(onBus: 0)
    playerNode.stop()
    audioEngine.stop()
    audioEngine.detach(playerNode)
    isCapturing = false
    // Flush any remaining accumulated audio
    sendQueue.async {
      if !self.accumulatedData.isEmpty {
        let chunk = self.accumulatedData
        self.accumulatedData = Data()
        self.onAudioCaptured?(chunk)
      }
    }
    removeObservers()
  }

  // MARK: - Audio Interruption & Route Change Handling

  private func setupInterruptionHandling() {
    interruptionObserver = NotificationCenter.default.addObserver(
      forName: AVAudioSession.interruptionNotification,
      object: AVAudioSession.sharedInstance(),
      queue: .main
    ) { [weak self] notification in
      guard let self,
            let userInfo = notification.userInfo,
            let typeValue = userInfo[AVAudioSessionInterruptionTypeKey] as? UInt,
            let type = AVAudioSession.InterruptionType(rawValue: typeValue)
      else { return }

      var shouldResume = false
      if type == .ended,
         let optionsValue = userInfo[AVAudioSessionInterruptionOptionKey] as? UInt {
        let options = AVAudioSession.InterruptionOptions(rawValue: optionsValue)
        shouldResume = options.contains(.shouldResume)
      }

      self.handleInterruption(type: type, shouldResume: shouldResume)
    }

    routeChangeObserver = NotificationCenter.default.addObserver(
      forName: AVAudioSession.routeChangeNotification,
      object: AVAudioSession.sharedInstance(),
      queue: .main
    ) { [weak self] notification in
      guard let self,
            let userInfo = notification.userInfo,
            let reasonValue = userInfo[AVAudioSessionRouteChangeReasonKey] as? UInt,
            let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue)
      else { return }

      self.handleRouteChange(reason: reason)
    }

    mediaServicesResetObserver = NotificationCenter.default.addObserver(
      forName: AVAudioSession.mediaServicesWereResetNotification,
      object: AVAudioSession.sharedInstance(),
      queue: .main
    ) { [weak self] _ in
      self?.attemptAudioReset()
    }
  }

  private func setupAppLifecycleObservers() {
    foregroundObserver = NotificationCenter.default.addObserver(
      forName: UIApplication.willEnterForegroundNotification,
      object: nil,
      queue: .main
    ) { [weak self] _ in
      guard let self else { return }
      NSLog("[Audio] App will enter foreground")
      if self.isCapturing && !self.audioEngine.isRunning {
        NSLog("[Audio] Audio engine stopped while backgrounded, attempting reset")
        self.attemptAudioReset()
      }
    }
  }

  private func handleInterruption(type: AVAudioSession.InterruptionType, shouldResume: Bool) {
    switch type {
    case .began:
      NSLog("[Audio] Audio interruption began (e.g. phone call)")
      wasCapturingBeforeInterruption = isCapturing
      if isCapturing {
        audioEngine.pause()
      }
    case .ended:
      NSLog("[Audio] Audio interruption ended (shouldResume=%@)", shouldResume ? "true" : "false")
      if wasCapturingBeforeInterruption {
        resumeAudioAfterInterruption()
      }
    @unknown default:
      break
    }
  }

  private func handleRouteChange(reason: AVAudioSession.RouteChangeReason) {
    switch reason {
    case .newDeviceAvailable:
      NSLog("[Audio] New audio device available")
    case .oldDeviceUnavailable:
      NSLog("[Audio] Audio device removed")
      if isCapturing {
        attemptAudioReset()
      }
    case .categoryChange, .override, .wakeFromSleep, .routeConfigurationChange:
      NSLog("[Audio] Audio route change: %d", reason.rawValue)
    default:
      break
    }
  }

  private func resumeAudioAfterInterruption() {
    NSLog("[Audio] Resuming audio after interruption")
    let audioSession = AVAudioSession.sharedInstance()
    do {
      try audioSession.setActive(true)
      try audioEngine.start()
      NSLog("[Audio] Audio resumed successfully")
    } catch {
      NSLog("[Audio] Failed to resume audio: %@", error.localizedDescription)
      attemptAudioReset()
    }
  }

  private func attemptAudioReset() {
    NSLog("[Audio] Attempting audio reset")
    let wasCapturing = isCapturing

    if audioEngine.isRunning {
      audioEngine.stop()
    }
    audioEngine.inputNode.removeTap(onBus: 0)
    isCapturing = false

    if wasCapturing {
      do {
        try setupAudioSession(useIPhoneMode: useIPhoneMode)
        try startCapture()
        NSLog("[Audio] Audio reset successful")
      } catch {
        NSLog("[Audio] Audio reset failed: %@", error.localizedDescription)
      }
    }
  }

  private func removeObservers() {
    if let observer = interruptionObserver {
      NotificationCenter.default.removeObserver(observer)
      interruptionObserver = nil
    }
    if let observer = routeChangeObserver {
      NotificationCenter.default.removeObserver(observer)
      routeChangeObserver = nil
    }
    if let observer = mediaServicesResetObserver {
      NotificationCenter.default.removeObserver(observer)
      mediaServicesResetObserver = nil
    }
    if let observer = foregroundObserver {
      NotificationCenter.default.removeObserver(observer)
      foregroundObserver = nil
    }
  }

  // MARK: - Private helpers

  private func computeRMS(_ buffer: AVAudioPCMBuffer) -> Float {
    let frameCount = Int(buffer.frameLength)
    guard frameCount > 0, let floatData = buffer.floatChannelData else { return 0 }
    var sumSquares: Float = 0
    for i in 0..<frameCount {
      let s = floatData[0][i]
      sumSquares += s * s
    }
    return sqrt(sumSquares / Float(frameCount))
  }

  private func float32BufferToInt16Data(_ buffer: AVAudioPCMBuffer) -> Data {
    let frameCount = Int(buffer.frameLength)
    guard frameCount > 0, let floatData = buffer.floatChannelData else { return Data() }
    var int16Array = [Int16](repeating: 0, count: frameCount)
    for i in 0..<frameCount {
      let sample = max(-1.0, min(1.0, floatData[0][i]))
      int16Array[i] = Int16(sample * Float(Int16.max))
    }
    return int16Array.withUnsafeBufferPointer { ptr in
      Data(buffer: ptr)
    }
  }

  private func convertBuffer(
    _ inputBuffer: AVAudioPCMBuffer,
    using converter: AVAudioConverter,
    targetFormat: AVAudioFormat
  ) -> AVAudioPCMBuffer? {
    let ratio = targetFormat.sampleRate / inputBuffer.format.sampleRate
    let outputFrameCount = UInt32(Double(inputBuffer.frameLength) * ratio)
    guard outputFrameCount > 0 else { return nil }

    guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputFrameCount) else {
      return nil
    }

    var error: NSError?
    var consumed = false
    let status = converter.convert(to: outputBuffer, error: &error) { _, outStatus in
      if consumed {
        outStatus.pointee = .noDataNow
        return nil
      }
      consumed = true
      outStatus.pointee = .haveData
      return inputBuffer
    }

    // Check conversion actually produced valid data
    if let error {
      NSLog("[Audio] Converter error: %@", error.localizedDescription)
      return nil
    }
    guard status == .haveData || status == .inputRanDry else {
      NSLog("[Audio] Converter status: %d (expected haveData/inputRanDry)", status.rawValue)
      return nil
    }
    guard outputBuffer.frameLength > 0 else {
      NSLog("[Audio] Converter produced 0 frames")
      return nil
    }

    return outputBuffer
  }
}

// MARK: - Speech To Text Fallback
@MainActor
class SpeechToTextManager: ObservableObject {
  @Published var isListening: Bool = false
  @Published var finalTranscribedText: String = ""
  @Published var partialTranscribedText: String = ""

  private var speechRecognizer: SFSpeechRecognizer?
  private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
  private var recognitionTask: SFSpeechRecognitionTask?
  private var silenceTimer: Timer?
  private var lastEngine: AVAudioEngine?

  init() {
    setupRecognizer()
  }

  private func setupRecognizer() {
    speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
  }

  func requestAuthorization(completion: @escaping (Bool) -> Void) {
    SFSpeechRecognizer.requestAuthorization { status in
      DispatchQueue.main.async {
        completion(status == .authorized)
      }
    }
  }

  func startProcessing(from engine: AVAudioEngine) throws {
    guard let recognizer = speechRecognizer, recognizer.isAvailable else {
      throw NSError(domain: "SpeechToText", code: 1, userInfo: [NSLocalizedDescriptionKey: "Speech recognizer not available."])
    }

    self.lastEngine = engine
    cancelProcessing()
    let request = SFSpeechAudioBufferRecognitionRequest()
    request.shouldReportPartialResults = true
    self.recognitionRequest = request

    recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
      guard let self = self else { return }
      Task { @MainActor in
        if let result = result {
          let text = result.bestTranscription.formattedString
          self.partialTranscribedText = text
          
          self.silenceTimer?.invalidate()
          self.silenceTimer = Timer.scheduledTimer(withTimeInterval: 1.5, repeats: false) { _ in
             Task { @MainActor [weak self] in
                 guard let self = self else { return }
                 let finalText = self.partialTranscribedText
                 self.partialTranscribedText = ""
                 if !finalText.isEmpty {
                     self.finalTranscribedText = finalText
                 }
                 // Restart the tap to cleanly bypass Apple's 60-second limit
                 if let engine = self.lastEngine {
                     self.cancelProcessing()
                     try? self.startProcessing(from: engine)
                 }
             }
          }
          
          if result.isFinal {
            self.finalTranscribedText = text
            self.partialTranscribedText = ""
          }
        }
        if error != nil {
          self.cancelProcessing()
        }
      }
    }

    self.isListening = true
  }

  func processBuffer(_ buffer: AVAudioPCMBuffer) {
    self.recognitionRequest?.append(buffer)
  }

  func cancelProcessing() {
    silenceTimer?.invalidate()
    silenceTimer = nil
    recognitionTask?.cancel()
    recognitionTask = nil
    recognitionRequest?.endAudio()
    recognitionRequest = nil
    isListening = false
  }
}
