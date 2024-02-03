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
  func eval(niter: Int) -> [Double] {
    var runtimes: [Double] = []

    for _ in 0..<niter {
        var runtime = 0.0
        
        let mixture = generateInputData(shape: [1, 2, 192], random: true)
        let embedding = generateInputData(shape: [1, 1, 256], random: true) // constant
        let conv_buf = generateInputData(shape: [1, 4, 2, 97], random: true)
        let deconv_buf = generateInputData(shape: [1, 64, 2, 97], random: true)
        let gridnet_bufs_buf0_K_buf = generateInputData(shape: [4, 49, 582], random: true)
        let gridnet_bufs_buf0_V_buf = generateInputData(shape: [4, 49, 1552], random: true)
        let gridnet_bufs_buf0_c0 = generateInputData(shape: [1, 97, 64], random: true)
        let gridnet_bufs_buf0_h0 = generateInputData(shape: [1, 97, 64], random: true)
        let gridnet_bufs_buf1_K_buf = generateInputData(shape: [4, 49, 582], random: true)
        let gridnet_bufs_buf1_V_buf = generateInputData(shape: [4, 49, 1552], random: true)
        let gridnet_bufs_buf1_c0 = generateInputData(shape: [1, 97, 64], random: true)
        let gridnet_bufs_buf1_h0 = generateInputData(shape: [1, 97, 64], random: true)
        let gridnet_bufs_buf2_K_buf = generateInputData(shape: [4, 49, 582], random: true)
        let gridnet_bufs_buf2_V_buf = generateInputData(shape: [4, 49, 1552], random: true)
        let gridnet_bufs_buf2_c0 = generateInputData(shape: [1, 97, 64], random: true)
        let gridnet_bufs_buf2_h0 = generateInputData(shape: [1, 97, 64], random: true)
        let istft_buf = generateInputData(shape: [1, 2, 194, 1], random: true)
        
        /*var filtered_output = shape.[1, 2, 128]
        var out_conv_buf = [1, 4, 2, 97]
        var out_deconv_buf = [1, 64, 2, 97]
        var out_gridnet_bufs_buf0_K_buf = [4, 49, 582]
        var out_gridnet_bufs_buf0_V_buf = [4, 49, 1552]
        var out_gridnet_bufs_buf0_c0 = [1, 97, 64]
        var out_gridnet_bufs_buf0_h0 = [1, 97, 64]
        var out_gridnet_bufs_buf1_K_buf = [4, 49, 582]
        var out_gridnet_bufs_buf1_V_buf = [4, 49, 1552]
        var out_gridnet_bufs_buf1_c0 = [1, 97, 64]
        var out_gridnet_bufs_buf1_h0 = [1, 97, 64]
        var out_gridnet_bufs_buf2_K_buf = [4, 49, 582]
        var out_gridnet_bufs_buf2_V_buf = [4, 49, 1552]
        var out_gridnet_bufs_buf2_c0 = [1, 97, 64]
        var out_gridnet_bufs_buf2_h0 = [1, 97, 64]
        var out_istft_buf = [1, 2, 194, 1]*/

        // Inference
        let startTime = DispatchTime.now()
        let _ = try! ortSession.run(
            withInputs: [
                "mixture": mixture,
                "embedding": embedding,
                "conv_buf": conv_buf,
                "deconv_buf": deconv_buf,
                "gridnet_bufs::buf0::K_buf": gridnet_bufs_buf0_K_buf,
                "gridnet_bufs::buf0::V_buf": gridnet_bufs_buf0_V_buf,
                "gridnet_bufs::buf0::c0": gridnet_bufs_buf0_c0,
                "gridnet_bufs::buf0::h0": gridnet_bufs_buf0_h0,
                "gridnet_bufs::buf1::K_buf": gridnet_bufs_buf1_K_buf,
                "gridnet_bufs::buf1::V_buf": gridnet_bufs_buf1_V_buf,
                "gridnet_bufs::buf1::c0": gridnet_bufs_buf1_c0,
                "gridnet_bufs::buf1::h0": gridnet_bufs_buf1_h0,
                "gridnet_bufs::buf2::K_buf": gridnet_bufs_buf2_K_buf,
                "gridnet_bufs::buf2::V_buf": gridnet_bufs_buf2_V_buf,
                "gridnet_bufs::buf2::c0": gridnet_bufs_buf2_c0,
                "gridnet_bufs::buf2::h0": gridnet_bufs_buf2_h0,
                "istft_buf": istft_buf],
            outputNames: ["filtered_output", "out::conv_buf", "out::deconv_buf", "out::gridnet_bufs::buf0::K_buf", "out::gridnet_bufs::buf0::V_buf", "out::gridnet_bufs::buf0::c0", "out::gridnet_bufs::buf0::h0", "out::gridnet_bufs::buf1::K_buf", "out::gridnet_bufs::buf1::V_buf", "out::gridnet_bufs::buf1::c0", "out::gridnet_bufs::buf1::h0", "out::gridnet_bufs::buf2::K_buf", "out::gridnet_bufs::buf2::V_buf", "out::gridnet_bufs::buf2::c0", "out::gridnet_bufs::buf2::h0", "out::istft_buf"],
            runOptions: nil)
        let endTime = DispatchTime.now()

        runtime += Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1e6
        runtimes.append(runtime)
    }

    return runtimes
  }

}
