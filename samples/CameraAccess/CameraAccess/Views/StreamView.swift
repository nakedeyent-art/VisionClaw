/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// StreamView.swift
//
// Main UI for video streaming from Meta wearable devices using the DAT SDK.
// This view demonstrates the complete streaming API: video streaming with real-time display, photo capture,
// and error handling. Extended with Gemini Live AI assistant and WebRTC live streaming integration.
//

import MWDATCore
import SwiftUI

struct StreamView: View {
  @ObservedObject var viewModel: StreamSessionViewModel
  @ObservedObject var wearablesVM: WearablesViewModel
  @ObservedObject var geminiVM: GeminiSessionViewModel
  @ObservedObject var webrtcVM: WebRTCSessionViewModel

  var body: some View {
    ZStack {
      // Black background for letterboxing/pillarboxing
      Color.black
        .edgesIgnoringSafeArea(.all)

      // Video backdrop: PiP when WebRTC connected, otherwise single local feed
      if webrtcVM.isActive && webrtcVM.connectionState == .connected {
        PiPVideoView(
          localFrame: viewModel.currentVideoFrame,
          remoteVideoTrack: webrtcVM.remoteVideoTrack,
          hasRemoteVideo: webrtcVM.hasRemoteVideo
        )
      } else if let videoFrame = viewModel.currentVideoFrame, viewModel.hasReceivedFirstFrame {
        GeometryReader { geometry in
          Image(uiImage: videoFrame)
            .resizable()
            .aspectRatio(contentMode: .fill)
            .frame(width: geometry.size.width, height: geometry.size.height)
            .clipped()
        }
        .edgesIgnoringSafeArea(.all)
      } else {
        ProgressView()
          .scaleEffect(1.5)
          .foregroundColor(.white)
      }

      // Gemini status overlay (top) + speaking indicator
      if geminiVM.isGeminiActive {
        VStack {
          GeminiStatusBar(geminiVM: geminiVM)
          Spacer()

          VStack(spacing: 8) {
            if !geminiVM.userTranscript.isEmpty || !geminiVM.aiTranscript.isEmpty {
              TranscriptView(
                userText: geminiVM.userTranscript,
                aiText: geminiVM.aiTranscript
              )
            }

            ToolCallStatusView(status: geminiVM.toolCallStatus)

            if geminiVM.isModelSpeaking {
              HStack(spacing: 8) {
                Image(systemName: "speaker.wave.2.fill")
                  .foregroundColor(.white)
                  .font(.system(size: 14))
                SpeakingIndicator()
              }
              .padding(.horizontal, 16)
              .padding(.vertical, 8)
              .background(Color.black.opacity(0.5))
              .cornerRadius(20)
            }
          }
          .padding(.bottom, 80)
        }
        .padding(.all, 24)
      }

      // WebRTC status overlay (top)
      if webrtcVM.isActive {
        VStack {
          WebRTCStatusBar(webrtcVM: webrtcVM)
          Spacer()
        }
        .padding(.all, 24)
      }

      // Bottom controls layer
      VStack(spacing: 12) {
        Spacer()
        // Text input bar (visible when Gemini is active)
        if geminiVM.isGeminiActive {
          TextInputBar(geminiVM: geminiVM)
        }
        ControlsView(viewModel: viewModel, geminiVM: geminiVM, webrtcVM: webrtcVM)
      }
      .padding(.all, 24)
    }
    .onDisappear {
      Task {
        if viewModel.streamingStatus != .stopped {
          await viewModel.stopSession()
        }
        if geminiVM.isGeminiActive {
          geminiVM.stopSession()
        }
        if webrtcVM.isActive {
          webrtcVM.stopSession()
        }
      }
    }
    // Show captured photos from DAT SDK in a preview sheet
    .sheet(isPresented: $viewModel.showPhotoPreview) {
      if let photo = viewModel.capturedPhoto {
        PhotoPreviewView(
          photo: photo,
          onDismiss: {
            viewModel.dismissPhotoPreview()
          }
        )
      }
    }
    // Gemini error alert
    .alert("AI Assistant", isPresented: Binding(
      get: { geminiVM.errorMessage != nil },
      set: { if !$0 { geminiVM.errorMessage = nil } }
    )) {
      Button("OK") { geminiVM.errorMessage = nil }
    } message: {
      Text(geminiVM.errorMessage ?? "")
    }
    // WebRTC error alert
    .alert("Live Stream", isPresented: Binding(
      get: { webrtcVM.errorMessage != nil },
      set: { if !$0 { webrtcVM.errorMessage = nil } }
    )) {
      Button("OK") { webrtcVM.errorMessage = nil }
    } message: {
      Text(webrtcVM.errorMessage ?? "")
    }
  }
}

// Extracted controls for clarity
struct ControlsView: View {
  @ObservedObject var viewModel: StreamSessionViewModel
  @ObservedObject var geminiVM: GeminiSessionViewModel
  @ObservedObject var webrtcVM: WebRTCSessionViewModel

  var body: some View {
    // Controls row
    HStack(spacing: 8) {
      CustomButton(
        title: "Stop streaming",
        style: .destructive,
        isDisabled: false
      ) {
        Task {
          await viewModel.stopSession()
        }
      }

      // Photo button (glasses mode only -- DAT SDK capture)
      if viewModel.streamingMode == .glasses {
        CircleButton(icon: "camera.fill", text: nil) {
          viewModel.capturePhoto()
        }
      }

      // Gemini AI button (disabled when WebRTC is active — audio conflict)
      CircleButton(
        icon: geminiVM.isGeminiActive ? "waveform.circle.fill" : "waveform.circle",
        text: "AI"
      ) {
        Task {
          if geminiVM.isGeminiActive {
            geminiVM.stopSession()
          } else {
            await geminiVM.startSession()
          }
        }
      }
      .opacity(webrtcVM.isActive ? 0.4 : 1.0)
      .disabled(webrtcVM.isActive)

      // WebRTC Live Stream button (disabled when Gemini is active — audio conflict)
      CircleButton(
        icon: webrtcVM.isActive
          ? "antenna.radiowaves.left.and.right.circle.fill"
          : "antenna.radiowaves.left.and.right.circle",
        text: "Live"
      ) {
        Task {
          if webrtcVM.isActive {
            webrtcVM.stopSession()
          } else {
            await webrtcVM.startSession()
          }
        }
      }
      .opacity(geminiVM.isGeminiActive ? 0.4 : 1.0)
      .disabled(geminiVM.isGeminiActive)
    }
  }
}

/// Text input bar for typing commands directly to Gemini when voice is unreliable.
struct TextInputBar: View {
  @ObservedObject var geminiVM: GeminiSessionViewModel
  @State private var textInput: String = ""
  @FocusState private var isFocused: Bool

  var body: some View {
    HStack(spacing: 8) {
      TextField("Type a command...", text: $textInput)
        .textFieldStyle(.plain)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .foregroundColor(.white)
        .focused($isFocused)
        .submitLabel(.send)
        .onSubmit { sendMessage() }

      Button(action: sendMessage) {
        Image(systemName: "arrow.up.circle.fill")
          .font(.system(size: 28))
          .foregroundColor(textInput.isEmpty ? .gray : .white)
      }
      .disabled(textInput.isEmpty)
    }
    .padding(.horizontal, 12)
    .padding(.vertical, 6)
    .background(Color.black.opacity(0.6))
    .cornerRadius(24)
    .overlay(
      RoundedRectangle(cornerRadius: 24)
        .stroke(Color.white.opacity(0.2), lineWidth: 1)
    )
  }

  private func sendMessage() {
    let message = textInput.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !message.isEmpty else { return }
    geminiVM.sendTextMessage(message)
    textInput = ""
    isFocused = false
  }
}
