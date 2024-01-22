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
    @State private var runtime = "--"

    private func compute_runtime(_ niter: Int) -> Double {
        return model.eval(niter: niter)
    }

    var body: some View {
        VStack {
            TextField("Enter a positive integer", value: $niter, formatter: NumberFormatter())
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .keyboardType(.numberPad)
            Button(action: {
                runtime = String(compute_runtime(niter))
            }) {
                Text("Compute runtime")
            }
            Text("Runtime avergaed over \(niter) iterations: \(runtime) ms")
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
