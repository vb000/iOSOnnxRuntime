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

    for _ in 0..<niter {
        let x = generateInputData(shape: [1, 2, 480], random: true)
        let label = generateInputData(shape: [1, 20])
        let ctx_buf_l = generateInputData(shape: [1, 2, 512, 256], random: true)
        let ctx_buf_r = generateInputData(shape: [1, 2, 512, 256], random: true)
        let out_buf_l = generateInputData(shape: [1, 1, 64, 4], random: true)
        let out_buf_r = generateInputData(shape: [1, 1, 64, 4], random: true)

        // Inference
        let startTime = DispatchTime.now()
        let _ = try! ortSession.run(
            withInputs: ["x": x,
                         "init_ctx_buf_l": ctx_buf_l,
                         "init_ctx_buf_r": ctx_buf_r,
                         "init_out_buf_l": out_buf_l,
                         "init_out_buf_r": out_buf_r],
            outputNames: ["filtered",
                          "out_buf_l",
                          "out_buf_r",
                         ],
            runOptions: nil)
        let endTime = DispatchTime.now()

        runtime += Double(
            endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1e6
    }

    return runtime / Double(niter)
  }

}
