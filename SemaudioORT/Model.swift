// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import Foundation
import onnxruntime_objc

class Model {
  private let ortEnv: ORTEnv
  private let ortSession: ORTSession

  enum ModelError: Error {
    case Error(_ message: String)
  }

  init() throws {
    let mainBundlePath = Bundle.main.bundleURL.path
    print("The absolute path of the main bundle is: \(mainBundlePath)")

    ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
    guard let modelPath = Bundle.main.path(forResource: "model", ofType: "ort") else {
      throw ModelError.Error("Failed to find model file.")
    }
    ortSession = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: nil)
  }

  // Function to generate ORT tensor of a given shape with the option of initializing
  // with either random values or zeros.
  func generateInputData(shape: [Int], random: Bool = false) -> ORTValue {
      let inputData: [Float]
      if random {
          inputData = (0..<shape.reduce(1, *)) .map { _ in Float.random(in: -1...1) }
      } else {
          inputData = [Float](repeating: 0.0, count: shape.reduce(1, *))
      }

      // Convert inputData to Bytes
      let inputDataBytes = Data (
          bytes: inputData,
          count: inputData.count * MemoryLayout<Float>.stride
      )

      let inputShape: [NSNumber] = shape.map { NSNumber(value: $0) }

      // try to create ORTValue and hanle error
      let input = try! ORTValue(
          tensorData: NSMutableData(data: inputDataBytes),
          elementType: ORTTensorElementDataType.float,
          shape: inputShape)
    
      return input
  }

  // Function to compute the runtime of the model in milliseconds averaged over
  // `niters`` iterations.
  func eval(niter: Int) -> Double {
    var runtime = 0.0
      var inference_times: [Double] = []
      
    for _ in 0..<niter {
        let random_input = generateInputData(shape: [1, 2, 80000], random: true)
        // Inference
        let startTime = DispatchTime.now()
        let _ = try! ortSession.run(
            withInputs: ["enrollment": random_input],
            outputNames: ["embeding"],
            runOptions: nil)
        let endTime = DispatchTime.now()

        let time_taken: Double = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1e6
        runtime += time_taken
        inference_times.append(time_taken)
    }
      
    print(inference_times)
    return runtime / Double(niter)
  }

}

