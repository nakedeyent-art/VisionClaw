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
  private var isLocateInFlight = false
  private var velocity: CGPoint = .zero
  private var lastRawPoint: CGPoint?
  private var lastUpdateTime: Date = .distantPast
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
    velocity = .zero
    lastRawPoint = nil
    isLocateInFlight = false
    matchCount = 0
    confidence = 0.0
    NSLog("[GazeControl] Session stopped")
  }

  // MARK: - Frame Processing

  func processFrame(_ image: UIImage) {
    guard isActive, !isLocateInFlight else { return }

    let now = Date()
    guard now.timeIntervalSince(lastSendTime) >= GazeConfig.gazeUpdateInterval else { return }
    lastSendTime = now

    guard let jpegData = image.jpegData(compressionQuality: GazeConfig.locateJpegQuality) else { return }

    isLocateInFlight = true

    Task {
      let result = await cursorBridge.locateGaze(imageData: jpegData)

      await MainActor.run {
        self.isLocateInFlight = false

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

  private func interpolate() {
    guard let current = smoothedPoint else { return }
    let speed = sqrt(velocity.x * velocity.x + velocity.y * velocity.y)
    guard speed > 2.0 else { return }  // Only interpolate if moving

    // Decay velocity over time
    let decay: CGFloat = 0.92
    velocity = CGPoint(x: velocity.x * decay, y: velocity.y * decay)

    let dt: CGFloat = 1.0 / 60.0
    let predicted = clampToScreen(CGPoint(
      x: current.x + velocity.x * dt,
      y: current.y + velocity.y * dt
    ))

    smoothedPoint = predicted
    gazeScreenPoint = predicted

    if isDragging {
      cursorBridge.mouseDragTo(predicted)
    } else {
      cursorBridge.moveCursor(to: predicted)
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

  private func applySmoothedPoint(_ raw: CGPoint) {
    let clamped = clampToScreen(raw)
    let now = Date()

    if let prev = smoothedPoint {
      // Adaptive alpha: faster response when moving fast
      let dx = clamped.x - prev.x
      let dy = clamped.y - prev.y
      let distance = sqrt(dx * dx + dy * dy)
      let alpha = min(0.6, max(0.15, distance / 500.0))

      let newPoint = CGPoint(
        x: prev.x + alpha * (clamped.x - prev.x),
        y: prev.y + alpha * (clamped.y - prev.y)
      )

      // Track velocity (pixels per second)
      let dt = now.timeIntervalSince(lastUpdateTime)
      if dt > 0 && dt < 1.0 {
        let vx = (newPoint.x - prev.x) / dt
        let vy = (newPoint.y - prev.y) / dt
        velocity = CGPoint(x: vx * 0.7 + velocity.x * 0.3, y: vy * 0.7 + velocity.y * 0.3)
      }

      smoothedPoint = newPoint
    } else {
      smoothedPoint = clamped
      velocity = .zero
    }

    lastUpdateTime = now
    lastRawPoint = clamped
    gazeScreenPoint = smoothedPoint

    guard let point = smoothedPoint else { return }

    if isDragging {
      cursorBridge.mouseDragTo(point)
    } else {
      cursorBridge.moveCursor(to: point)
    }
  }
}
