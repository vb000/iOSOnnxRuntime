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

    private func compute_runtime(_ niter: Int) async -> Double {
        return model.eval(niter: niter)
    }

    var body: some View {
        VStack {
            TextField("Enter a positive integer", value: $niter, formatter: NumberFormatter())
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .keyboardType(.numberPad)
            Button(action: {
                // Using Task to perform asynchronous operation
                Task {
                    let result = await compute_runtime(niter)
                    // Update the UI on the main thread
                    DispatchQueue.main.async {
                        runtime = String(result)
                    }
                }
            }) {
                Text("Compute runtime")
            }
            Text("Runtime averaged over \(niter) iterations: \(runtime) ms")
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
