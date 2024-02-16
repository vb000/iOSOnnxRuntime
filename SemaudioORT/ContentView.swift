//
//  ContentView.swift
//  Hello
//
//  Created by Bandhav Veluri on 3/14/23.
//

import SwiftUI

struct ContentView: View {
    private let model = try! EnrollModel()
    
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("TSH enroll function")
            Button(action: {
                Task {
                    model.eval()
                }
            }) {
                Text("runEnroll")
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}

