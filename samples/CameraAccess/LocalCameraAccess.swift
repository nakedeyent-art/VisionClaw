import Foundation
import UIKit
import Combine
import AVFoundation

/// `LocalCameraAccess` acts as the native iOS bridge to your iPhone's completely offline LocalAI engine.
/// It bypasses Google's WebRTC infrastructure in favor of a low-latency, frame-by-frame snapshot analysis
/// using the standard OpenAI/LLaVA REST schema pointing directly to `localhost`.
class LocalCameraAccess: ObservableObject {
    static let shared = LocalCameraAccess()
    
    // Core UI state variables
    @Published var isProcessing: Bool = false
    @Published var lastResponse: String = "Awaiting first frame..."
    @Published var connectionStatus: String = "Disconnected"
    
    private var lastProcessTime: Date = Date.distantPast
    // We throttle to 1 frame every 1.5 seconds to balance real-time feel with battery/heat constraints
    private let processingInterval: TimeInterval = 1.5
    
    // The endpoint where your offline iPhone LocalAI server handles image requests
    private let localApiUrl = "http://localhost:8080/v1/chat/completions"
    
    /// Called by the AVFoundation camera delegate to pass the visual feed into the offline AI.
    func processFrame(_ image: UIImage, prompt: String = "Describe what you see in one quick, analytical sentence.") {
        let now = Date()
        
        // Prevent bombarding the local neural engine. Wait until interval passes and previous frame finishes returning
        guard now.timeIntervalSince(lastProcessTime) >= processingInterval, !isProcessing else {
            return
        }
        
        lastProcessTime = now
        isProcessing = true
        connectionStatus = "Analyzing Local Frame..."
        
        // Compress image severely to keep base64 payload tiny and fast for the local API
        guard let imageData = image.jpegData(compressionQuality: 0.4) else {
            self.isProcessing = false
            return
        }
        let base64Image = imageData.base64EncodedString()
        
        postToLocalAI(base64Image: base64Image, prompt: prompt)
    }
    
    private func postToLocalAI(base64Image: String, prompt: String) {
        guard let url = URL(string: localApiUrl) else { return }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Standard payload format expected by LocalAI running LLaVA or Qwen-VL Vision Models
        let payload: [String: Any] = [
            "model": "gpt-4-vision-preview", // LocalAI typically aliases vision models to this handler
            "messages": [
                [
                    "role": "user",
                    "content": [
                        ["type": "text", "text": prompt],
                        ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64,\(base64Image)"]]
                    ]
                ]
            ],
            "max_tokens": 75,
            "stream": false
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        } catch {
            DispatchQueue.main.async {
                self.isProcessing = false
                self.connectionStatus = "Error: Payload Construction Failed"
            }
            return
        }
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                self.isProcessing = false
                
                // Handle total failure (LocalAI is off or crashed)
                if let error = error {
                    self.connectionStatus = "Error: LocalAI Unreachable"
                    self.lastResponse = "Ensure LocalAI is open and running on port 8080."
                    print("LocalAI Connection Error: \(error.localizedDescription)")
                    return
                }
                
                self.connectionStatus = "Connected (Localhost)"
                
                // Safely parse the returning JSON analysis and post it to the app's UI
                if let data = data {
                    do {
                        if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                           let choices = json["choices"] as? [[String: Any]],
                           let firstChoice = choices.first,
                           let message = firstChoice["message"] as? [String: Any],
                           let content = message["content"] as? String {
                            self.lastResponse = content
                        } else {
                            self.lastResponse = "Unreadable response pattern from Local Engine."
                        }
                    } catch {
                        self.lastResponse = "JSON Parsing failure occurred."
                    }
                }
            }
        }
        task.resume()
    }
}
