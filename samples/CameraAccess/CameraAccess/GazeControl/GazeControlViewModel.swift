import Foundation
import SwiftUI

enum GazeMode: String {
  case connecting
  case tracking
  case noMatch
  case dragging
}

@MainActor
class GazeControlViewModel: ObservableObject {
  @Published var isActive = false
  @Published var mode: GazeMode = .connecting
  @Published var gazeScreenPoint: CGPoint?
  @Published var isDragging = false
  @Published var errorMessage: String?
  @Published var matchCount: Int = 0
  @Published var confidence: Double = 0.0

  let cursorBridge = CursorControlBridge()

  private var lastSendTime: Date = .distantPast
  private var smoothedPoint: CGPoint?
  private var targetPoint: CGPoint?  // Where we're heading
  private var locateSeq: UInt64 = 0  // Pipelining: only accept latest result
  private var lastAppliedSeq: UInt64 = 0
  private var inFlightCount: Int = 0
  private var interpolationTimer: Timer?

  // MARK: - Session Control

  func startSession() async {
    isActive = true
    mode = .connecting
    gazeScreenPoint = nil
    smoothedPoint = nil
    matchCount = 0
    confidence = 0.0

    await cursorBridge.checkConnection()

    if cursorBridge.connectionState != .connected {
      errorMessage = "Cannot reach cursor server at \(GazeConfig.cursorServerBaseURL)"
      isActive = false
      return
    }

    mode = .tracking
    startInterpolation()
    NSLog("[GazeControl] Session started (server-side matching)")
  }

  func stopSession() {
    if isDragging, let pt = smoothedPoint {
      cursorBridge.mouseUp(at: pt)
      isDragging = false
    }
    stopInterpolation()
    isActive = false
    mode = .connecting
    gazeScreenPoint = nil
    smoothedPoint = nil
    targetPoint = nil
    locateSeq = 0
    lastAppliedSeq = 0
    inFlightCount = 0
    matchCount = 0
    confidence = 0.0
    NSLog("[GazeControl] Session stopped")
  }

  // MARK: - Frame Processing

  func processFrame(_ image: UIImage) {
    guard isActive, inFlightCount < 2 else { return }

    let now = Date()
    guard now.timeIntervalSince(lastSendTime) >= GazeConfig.gazeUpdateInterval else { return }
    lastSendTime = now

    guard let jpegData = image.jpegData(compressionQuality: GazeConfig.locateJpegQuality) else { return }

    locateSeq += 1
    let seq = locateSeq
    inFlightCount += 1

    Task {
      let result = await cursorBridge.locateGaze(imageData: jpegData)

      await MainActor.run {
        self.inFlightCount -= 1

        // Discard stale responses — only apply the latest
        guard seq > self.lastAppliedSeq else { return }
        self.lastAppliedSeq = seq

        guard let result = result else {
          self.mode = self.isDragging ? .dragging : .noMatch
          return
        }

        self.matchCount = result.matchCount
        self.confidence = result.confidence

        if let point = result.point {
          self.mode = self.isDragging ? .dragging : .tracking
          self.applySmoothedPoint(point)
        } else {
          if !self.isDragging {
            self.mode = .noMatch
          }
        }
      }
    }
  }

  // MARK: - Drag Mode

  func toggleDrag() {
    guard mode == .tracking || mode == .dragging else { return }

    if isDragging {
      if let pt = smoothedPoint {
        cursorBridge.mouseUp(at: pt)
      }
      isDragging = false
      mode = .tracking
      NSLog("[GazeControl] Drag released")
    } else {
      if let pt = smoothedPoint {
        cursorBridge.mouseDown(at: pt)
        isDragging = true
        mode = .dragging
        NSLog("[GazeControl] Drag started at %.0f, %.0f", pt.x, pt.y)
      }
    }
  }

  func triggerClick() {
    guard mode == .tracking, let pt = smoothedPoint else { return }
    cursorBridge.click(at: pt)
    NSLog("[GazeControl] Click at %.0f, %.0f", pt.x, pt.y)
  }

  // MARK: - Interpolation

  private func startInterpolation() {
    interpolationTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
      Task { @MainActor in
        self?.interpolate()
      }
    }
  }

  private func stopInterpolation() {
    interpolationTimer?.invalidate()
    interpolationTimer = nil
  }

  /// 60fps lerp: smoothly move cursor toward target
  private func interpolate() {
    guard let target = targetPoint, let current = smoothedPoint else { return }

    let dx = target.x - current.x
    let dy = target.y - current.y
    let dist = sqrt(dx * dx + dy * dy)

    guard dist > 0.5 else { return }  // Close enough, skip

    // Lerp factor per frame: 30% per frame at 60fps — responsive but smooth
    let lerp: CGFloat = 0.3
    let next = CGPoint(
      x: current.x + dx * lerp,
      y: current.y + dy * lerp
    )

    smoothedPoint = next
    gazeScreenPoint = next

    if isDragging {
      cursorBridge.mouseDragTo(next)
    } else {
      cursorBridge.moveCursor(to: next)
    }
  }

  // MARK: - Internal

  private func clampToScreen(_ point: CGPoint) -> CGPoint {
    let screenSize = cursorBridge.remoteScreenSize ?? CGSize(width: 1920, height: 1080)
    let origin = cursorBridge.remoteScreenOrigin
    return CGPoint(
      x: max(origin.x, min(origin.x + screenSize.width, point.x)),
      y: max(origin.y, min(origin.y + screenSize.height, point.y))
    )
  }

  /// Server returned a new point — set it as the target, timer will lerp toward it
  private func applySmoothedPoint(_ raw: CGPoint) {
    let clamped = clampToScreen(raw)
    targetPoint = clamped

    // First point: jump directly
    if smoothedPoint == nil {
      smoothedPoint = clamped
      gazeScreenPoint = clamped
      cursorBridge.moveCursor(to: clamped)
    }
  }
}
