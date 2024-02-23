//
//  ContentView.swift
//  Hello
//
//  Created by Bandhav Veluri on 3/14/23.
//

import SwiftUI
import onnxruntime_objc

struct ContentView: View {
  private let model = try! Model()
  @State private var model_running = false

  var body: some View {
    VStack {
      Button(action: {
        Task {
          
        }
      }) {
        Text("Enroll Placeholder")
      }
      Button(action: {
        Task {
          await toggle_tsh()
          model_running = !model_running
        }
      }) {
        if (model_running) {
          Text("Turn off TSH")
        } else {
          Text("Begin TSH")
        }
      }
    }.buttonStyle(.bordered)
  }
  
  private func toggle_tsh() async -> Void {
    
  }
}

struct ContentView_Previews: PreviewProvider {
  static var previews: some View {
    ContentView()
  }
}
