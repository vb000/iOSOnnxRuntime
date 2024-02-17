//
//  ContentView.swift
//  Hello
//
//  Created by Bandhav Veluri on 3/14/23.
//

import SwiftUI

struct ContentView: View {
  private let model = try! Model()
  @State private var niter = 1
  @State private var summary = "Waiting for input"
  @State private var runtimes = "--"
  @State private var average = "--"
  @State private var std = "--"
  @State private var computing = false

  // runs the model and updates state to display results
  private func compute_runtimes(_ niter: Int) async -> Void {
      let times: [Double] = model.eval(niter: niter)
      summary = "For \(niter) iterations:"
      runtimes = "[" + times.map { String($0) }.joined(separator: ", ") + "]"
      average = String(times.avg())
      std = String(times.std())
      print(runtimes)
  }
  
  private func test_method() async -> Void {
    let inputs = model.getInputs()
    let outputs = model.runModel(inputs: inputs)
    print(outputs)
  }

    var body: some View {
        VStack {
            TextField("Enter a positive integer", value: $niter, formatter: NumberFormatter())
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .keyboardType(.numberPad)
            Button(action: {
                computing = true
                summary = "Model Running"
                Task {
                    await compute_runtimes(niter)
                    computing = false
                }
            }) {
                if (computing) {
                    Text("Computing")
                } else {
                    Text("Compute Runtime")
                }
            }.disabled(computing)
            Text("\(summary)")
            Text("Average: \(average) ms")
            Text("STD: \(std) ms")
            Text("Runtimes: \(runtimes)")
            Button(action: {
                computing = true
                summary = "Model Testing"
                Task {
                    await test_method()
                    computing = false
                }
            }) {
                if (computing) {
                    Text("Testing")
                } else {
                    Text("Test")
                }
            }.disabled(computing)
        }
        .buttonStyle(.bordered)
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
