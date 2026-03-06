import Foundation

enum CursorServerConnectionState: Equatable {
  case disconnected
  case checking
  case connected
  case unreachable(String)
}

@MainActor
class CursorControlBridge: ObservableObject {
  @Published var connectionState: CursorServerConnectionState = .disconnected
  @Published var remoteScreenSize: CGSize?
  @Published var remoteScreenOrigin: CGPoint = .zero  // Can be negative for multi-monitor

  private let session: URLSession
  private let pingSession: URLSession

  init() {
    let config = URLSessionConfiguration.default
    config.timeoutIntervalForRequest = 2  // Fast timeout for real-time control
    self.session = URLSession(configuration: config)

    let pingConfig = URLSessionConfiguration.default
    pingConfig.timeoutIntervalForRequest = 5
    self.pingSession = URLSession(configuration: pingConfig)
  }

  // MARK: - Connection

  func checkConnection() async {
    guard GazeConfig.isCursorServerConfigured else {
      connectionState = .disconnected
      return
    }
    connectionState = .checking
    guard let url = URL(string: "\(GazeConfig.cursorServerBaseURL)/health") else {
      connectionState = .unreachable("Invalid URL")
      return
    }
    do {
      let (data, response) = try await pingSession.data(for: URLRequest(url: url))
      if let http = response as? HTTPURLResponse, http.statusCode == 200 {
        connectionState = .connected
        NSLog("[GazeCursor] Server reachable")
        await fetchScreenSize()
      } else {
        connectionState = .unreachable("Unexpected response")
      }
    } catch {
      connectionState = .unreachable(error.localizedDescription)
      NSLog("[GazeCursor] Server unreachable: %@", error.localizedDescription)
    }
  }

  func fetchScreenSize() async {
    guard let url = URL(string: "\(GazeConfig.cursorServerBaseURL)/screen") else { return }
    do {
      let (data, _) = try await session.data(for: URLRequest(url: url))
      if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
         let w = json["width"] as? Double,
         let h = json["height"] as? Double {
        remoteScreenSize = CGSize(width: w, height: h)
        let ox = json["origin_x"] as? Double ?? 0
        let oy = json["origin_y"] as? Double ?? 0
        remoteScreenOrigin = CGPoint(x: ox, y: oy)
        NSLog("[GazeCursor] Screen size: %.0fx%.0f origin: (%.0f, %.0f)", w, h, ox, oy)
      }
    } catch {
      NSLog("[GazeCursor] Failed to get screen size: %@", error.localizedDescription)
    }
  }

  // MARK: - Cursor Control (fire-and-forget)

  func moveCursor(to point: CGPoint) {
    sendCommand("move", body: ["x": point.x, "y": point.y])
  }

  func click(at point: CGPoint) {
    sendCommand("click", body: ["x": point.x, "y": point.y])
  }

  func mouseDown(at point: CGPoint) {
    sendCommand("mouse_down", body: ["x": point.x, "y": point.y])
  }

  func mouseDragTo(_ point: CGPoint) {
    sendCommand("mouse_drag_to", body: ["x": point.x, "y": point.y])
  }

  func mouseUp(at point: CGPoint) {
    sendCommand("mouse_up", body: ["x": point.x, "y": point.y])
  }

  // MARK: - Calibration

  struct CalibrationStatus {
    let calibrated: Bool
    let anchors: Int
    let calibrating: Bool
    let calIndex: Int
  }

  struct CalibrationPoint {
    let pointIndex: Int
    let totalPoints: Int
    let screenX: Double
    let screenY: Double
    let status: String?  // "captured", "all_captured", or nil (start)
  }

  func fetchCalibrationStatus() async -> CalibrationStatus? {
    guard let url = URL(string: "\(GazeConfig.cursorServerBaseURL)/calibrate/status") else { return nil }
    do {
      let (data, response) = try await session.data(for: URLRequest(url: url))
      guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { return nil }
      guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
      return CalibrationStatus(
        calibrated: json["calibrated"] as? Bool ?? false,
        anchors: json["anchors"] as? Int ?? 0,
        calibrating: json["calibrating"] as? Bool ?? false,
        calIndex: json["cal_index"] as? Int ?? 0
      )
    } catch {
      return nil
    }
  }

  func startCalibration() async -> CalibrationPoint? {
    guard let url = URL(string: "\(GazeConfig.cursorServerBaseURL)/calibrate/start") else { return nil }
    var req = URLRequest(url: url)
    req.httpMethod = "POST"
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    do {
      let (data, response) = try await session.data(for: req)
      guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { return nil }
      guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
      return CalibrationPoint(
        pointIndex: json["point_index"] as? Int ?? 0,
        totalPoints: json["total_points"] as? Int ?? 9,
        screenX: json["screen_x"] as? Double ?? 0,
        screenY: json["screen_y"] as? Double ?? 0,
        status: nil
      )
    } catch {
      return nil
    }
  }

  func captureCalibrationFrame(imageData: Data) async -> CalibrationPoint? {
    guard let url = URL(string: "\(GazeConfig.cursorServerBaseURL)/calibrate/capture") else { return nil }
    var req = URLRequest(url: url)
    req.httpMethod = "POST"
    req.setValue("image/jpeg", forHTTPHeaderField: "Content-Type")
    req.httpBody = imageData
    do {
      let (data, response) = try await session.data(for: req)
      guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { return nil }
      guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
      let status = json["status"] as? String
      return CalibrationPoint(
        pointIndex: json["point_index"] as? Int ?? 0,
        totalPoints: json["total_points"] as? Int ?? 9,
        screenX: json["screen_x"] as? Double ?? 0,
        screenY: json["screen_y"] as? Double ?? 0,
        status: status
      )
    } catch {
      return nil
    }
  }

  func finishCalibration() async -> Bool {
    guard let url = URL(string: "\(GazeConfig.cursorServerBaseURL)/calibrate/finish") else { return false }
    var req = URLRequest(url: url)
    req.httpMethod = "POST"
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    do {
      let (data, response) = try await session.data(for: req)
      guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { return false }
      guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let status = json["status"] as? String, status == "ok"
      else { return false }
      return true
    } catch {
      return false
    }
  }

  // MARK: - Locate (server-side matching)

  struct LocateResult {
    let status: String      // "ok" or "no_match"
    let point: CGPoint?     // screen coordinate (nil when no_match)
    let matchCount: Int
    let confidence: Double
  }

  /// POST a camera frame JPEG to /locate and get back screen coordinates.
  func locateGaze(imageData: Data) async -> LocateResult? {
    guard let url = URL(string: "\(GazeConfig.cursorServerBaseURL)/locate") else { return nil }

    var req = URLRequest(url: url)
    req.httpMethod = "POST"
    req.setValue("image/jpeg", forHTTPHeaderField: "Content-Type")
    req.httpBody = imageData

    do {
      let (data, response) = try await session.data(for: req)
      guard let http = response as? HTTPURLResponse, http.statusCode == 200 else { return nil }

      guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let status = json["status"] as? String
      else { return nil }

      let point: CGPoint?
      if status == "ok", let x = json["x"] as? Double, let y = json["y"] as? Double {
        point = CGPoint(x: x, y: y)
      } else {
        point = nil
      }

      return LocateResult(
        status: status,
        point: point,
        matchCount: json["matches"] as? Int ?? 0,
        confidence: json["confidence"] as? Double ?? 0.0
      )
    } catch {
      return nil
    }
  }

  // MARK: - Internal

  private func sendCommand(_ endpoint: String, body: [String: Any]) {
    guard let url = URL(string: "\(GazeConfig.cursorServerBaseURL)/\(endpoint)") else { return }
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try? JSONSerialization.data(withJSONObject: body)

    // Fire-and-forget: don't block the frame pipeline
    Task.detached { [session] in
      _ = try? await session.data(for: request)
    }
  }
}
