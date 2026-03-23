import AVFoundation
import Foundation
import UIKit

class AudioManager {
  var onAudioCaptured: ((Data) -> Void)?

  private let audioEngine = AVAudioEngine()
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
    // voiceChat: aggressive echo cancellation (mic + speaker co-located on phone)
    // videoChat: mild AEC (mic on glasses, speaker on glasses)
    // When Speaker Output is ON, speaker is on phone so always use voiceChat AEC
    let forceSpeaker = SettingsManager.shared.speakerOutputEnabled
    if useIPhoneMode || forceSpeaker {
      try session.setCategory(
        .playAndRecord,
        mode: .voiceChat,
        options: [.defaultToSpeaker, .allowBluetooth, .allowBluetoothA2DP, .mixWithOthers]
      )
    } else {
      try session.setCategory(
        .playAndRecord,
        mode: .voiceChat,
        options: [.defaultToSpeaker, .allowBluetooth, .allowBluetoothA2DP, .mixWithOthers]
      )
    }

    // CRITICAL: Force the built-in microphone as the preferred input.
    // When Bluetooth devices (Ray-Ban Metas, AirPods) are connected, iOS may
    // route audio input through a garbled HFP channel, producing noise.
    // The video comes from glasses via MWDAT, but audio capture must use
    // the iPhone's built-in mic for clean 16kHz PCM.
    if let availableInputs = session.availableInputs {
      NSLog("[Audio] Available inputs: %@", availableInputs.map { "\($0.portName) (\($0.portType.rawValue))" }.joined(separator: ", "))
      if let builtInMic = availableInputs.first(where: { $0.portType == .builtInMic }) {
        try session.setPreferredInput(builtInMic)
        NSLog("[Audio] ✅ Forced preferred input: %@ (%@)", builtInMic.portName, builtInMic.portType.rawValue)
      } else {
        NSLog("[Audio] ⚠️ No built-in mic found in available inputs!")
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
    NSLog("[Audio] Session mode: %@", useIPhoneMode ? "voiceChat (iPhone)" : "voiceChat (glasses)")

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

    // The target format Gemini expects: 16kHz Float32 Mono
    let geminiInputFormat = AVAudioFormat(
      commonFormat: .pcmFormatFloat32,
      sampleRate: GeminiConfig.inputAudioSampleRate,
      channels: GeminiConfig.audioChannels,
      interleaved: false
    )!

    // ──────────────────────────────────────────────────────────────────────
    //  MIXER NODE APPROACH (Apple's recommended pattern for format conversion)
    //
    //  Instead of manually using AVAudioConverter (which silently produces
    //  garbled output on certain hardware configs like iPhone 16 Pro Max),
    //  we insert a dedicated mixer node between the input and our tap.
    //  The mixer handles sample rate + channel conversion internally using
    //  Apple's DSP — this is the standard, reliable iOS pattern.
    //
    //  Input Node (48kHz stereo) → Mixer Node → Tap (16kHz mono)
    // ──────────────────────────────────────────────────────────────────────

    let formatConverterMixer = AVAudioMixerNode()
    audioEngine.attach(formatConverterMixer)

    // Connect input → mixer in the input's native format
    audioEngine.connect(inputNode, to: formatConverterMixer, format: inputNativeFormat)

    NSLog("[Audio] Mixer pipeline: %.0fHz %dch → %.0fHz %dch",
          inputNativeFormat.sampleRate, inputNativeFormat.channelCount,
          geminiInputFormat.sampleRate, geminiInputFormat.channelCount)

    sendQueue.async { self.accumulatedData = Data() }

    var tapCount = 0
    // Tap the mixer in our TARGET format — the mixer does the conversion!
    formatConverterMixer.installTap(onBus: 0, bufferSize: 4096, format: geminiInputFormat) { [weak self] buffer, _ in
      guard let self else { return }

      tapCount += 1
      // Buffer is already 16kHz Float32 Mono — just convert to Int16
      let pcmData = self.float32BufferToInt16Data(buffer)

      if tapCount <= 3 {
        NSLog("[Audio] Tap #%d: %d frames, %d bytes (format: %.0fHz %dch %@)",
              tapCount, buffer.frameLength, pcmData.count,
              buffer.format.sampleRate, buffer.format.channelCount,
              buffer.format.commonFormat == .pcmFormatFloat32 ? "Float32" : "Other")
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
                  chunk.count, chunk.count / 32)  // 16kHz * 2 bytes = 32 bytes/ms
          }
          self.onAudioCaptured?(chunk)
        }
      }
    }

    try audioEngine.start()
    playerNode.play()
    isCapturing = true
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
