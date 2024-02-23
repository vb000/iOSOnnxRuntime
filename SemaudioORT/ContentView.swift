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
  @State private var info: String = "Click something"

  var body: some View {
    VStack {
      // display info for the users
      Text("\(self.info)")
      
      // enrolls
      // not implemented yet
      Button(action: {
        Task {
          self.info = "Enroll functionality not implemented yet."
        }
      }) {
        Text("Enroll Placeholder")
      }.disabled(self.model_running)
      
      // turns on/off TSH
      Button(action: {
        Task {
          await toggle_tsh()
        }
      }) {
        if (self.model_running) {
          Text("Turn off TSH")
        } else {
          Text("Begin TSH")
        }
      }
      
      // Runs the model n times and prints runtime information
      Button(action: {
        Task {
          self.model_running = true
          await compute_runtimes(n: 100)
          self.model_running = false
        }
      }) {
        Text("Test Runtimes")
      }.disabled(self.model_running)
      
      // tests output when all inputs are vectors of 0s
      Button(action: {
        Task {
          self.model_running = true
          await testOutput(mixture: generateORTValue(shape: [1, 2, 192], random: false))
          self.model_running = false
        }
      }) {
        Text("Test Output")
      }.disabled(self.model_running)
      
    }.buttonStyle(.bordered)
  }
  
  // not implemented, will eventually begin/end TSH
  private func toggle_tsh() async -> Void {
    self.model_running = !self.model_running
    self.info = "TSH not implemented yet."
  }
  
  // runs model.infer() to get runtimes
  private func compute_runtimes(n: Int) async -> Void {
    // run infer() n times
    var times: [Double] = []
    model.resetState()
    for _ in 0..<n {
      let mixture = generateORTValue(shape: [1, 2, 192], random: true)
      let startTime = DispatchTime.now()
      var output = self.model.infer(mixture: mixture)
      let endTime = DispatchTime.now()
      let runtime = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1e6
      times.append(runtime)
    }
    // get data and print
    let average_string: String = "Average: " + String(times.avg())
    let std_string: String = "STD: " + String(times.std())
    print(times)
    print(average_string)
    print(std_string)
    self.info = "Ran model.infer() " + String(n) + " times.\n"
        + average_string + "ms\n"
        + std_string + "ms"
  }
  
  // runs model with state of 0s on mixture, prints the output
  private func testOutput(mixture: ORTValue) async -> Void {
    model.resetState()
    let output = model.infer(mixture: mixture)
    print(ORTValueToArray(input: output))
    self.info = "Output float array printed to console."
  }
  
}

struct ContentView_Previews: PreviewProvider {
  static var previews: some View {
    ContentView()
  }
}

// helpers for calculating mean and std
extension Array where Element: FloatingPoint {
    func sum() -> Element {
        return self.reduce(0, +)
    }

    func avg() -> Element {
        return self.sum() / Element(self.count)
    }

    func std() -> Element {
        let mean = self.avg()
        let v = self.reduce(0, { $0 + ($1-mean)*($1-mean) })
        return sqrt(v / (Element(self.count) - 1))
    }
}
