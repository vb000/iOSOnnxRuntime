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
  @State private var average = "__"
  @State private var niter = 1
  @State private var out = 1
  @State private var enrolling = false

  private func run_tsh(){
    model_running = !model_running
    DispatchQueue.global(qos: .background).async {
      while (model_running) {
        // When other team is done somehow make the input await for the 8ms mixture of real data then proceed
        let input = Model.generateFakeInput()
        let output = model.infer(mixture: input)
        out = out + 1
        //Here somehow output this sound
      }

    }
  }
  
  private func run_enrollment() {
    if (!enrolling) {
      enrolling = true
      DispatchQueue.global(qos: .background).async {
        do {
          sleep(4)
        }
        let enrollment = ""
  //      model.set_embed(embed: enrollment)
        enrolling = false
      }
    }
  }
  
  var body: some View {
    VStack {
      Button(action: {
        Task {
         run_enrollment()
        }
      }) {
        if (enrolling) {
          Text("Enrolling")
        } else {
          Text("Enroll")
        }
      }
      Button(action: {
        Task {
          run_tsh()
        }
      }) {
        if (model_running) {
          Text("Turn off TSH")
        } else {
          Text("Begin TSH")
        }
      }
      Text("Output: \(out)")
    }.buttonStyle(.bordered)
  }
}

struct ContentView_Previews: PreviewProvider {
  static var previews: some View {
    ContentView()
  }
}
