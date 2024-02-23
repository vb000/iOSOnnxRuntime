// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import Foundation
import onnxruntime_objc

class Model {
  private let ortEnv: ORTEnv
  private let ortSession: ORTSession
  private var buffers: Dictionary<String, ORTValue>

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
    
    buffers = initializeInputs()
  }
  
  // Sets enrollment to the enrollment passed in
  func set_embed(embed: ORTValue) -> Void {
    buffers["embedding"] = embed
  }
  
  // the important method
  // returns the filtered output [1, 2, 128] when the current buffers are run through the model, and updates the buffers accordingly
  func infer(mixture: ORTValue) -> ORTValue {
    // update mixture
    buffers["mixture"] = mixture
    // run the model
    let outputs = try! ortSession.run(
        withInputs: buffers,
        outputNames: ["filtered_output", "out::conv_buf", "out::deconv_buf", "out::gridnet_bufs::buf0::K_buf", "out::gridnet_bufs::buf0::V_buf", "out::gridnet_bufs::buf0::c0", "out::gridnet_bufs::buf0::h0", "out::gridnet_bufs::buf1::K_buf", "out::gridnet_bufs::buf1::V_buf", "out::gridnet_bufs::buf1::c0", "out::gridnet_bufs::buf1::h0", "out::gridnet_bufs::buf2::K_buf", "out::gridnet_bufs::buf2::V_buf", "out::gridnet_bufs::buf2::c0", "out::gridnet_bufs::buf2::h0", "out::istft_buf"],
        runOptions: nil)
    // update internal buffers
    for name_ortvalue_pair in outputs {
      if name_ortvalue_pair.key == "filtered_output" {
        continue
      }
      var buffer_name: String = name_ortvalue_pair.key.replacingOccurrences(of: "out::", with: "")
      buffers[buffer_name] = name_ortvalue_pair.value
    }
    
    // return
    if let output = outputs["filtered_output"] {
      return output
    } else {
      return generateORTValue(shape: [1, 2, 128], random: false)
    }
  }
  
  func get_avg_runtimes(niter: Int) -> Double {
    var mixtures: [ORTValue] = []
    
    for _ in 0..<niter {
      mixtures.append(generateORTValue(shape: [1, 2, 192], random: true))
    }
    
    let startTime = DispatchTime.now()
    
    
    for mixture in mixtures{
      let _ = infer(mixture: mixture)
    }
    
    let endTime = DispatchTime.now()
    return (Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1e6) / Double(niter)
  }

  static func generateFakeInput() -> ORTValue {
   return generateORTValue(shape: [1, 2, 192], random: true)
  }
  
}

// STATIC HELPER FUNCTIONS

// static function to get zeroed inputs for the model
func initializeInputs() -> Dictionary<String, ORTValue> {
  let mixture = generateORTValue(shape: [1, 2, 192], random: false)
  let embedding = generateORTValue(shape: [1, 1, 256], random: false) // constant
  let conv_buf = generateORTValue(shape: [1, 4, 2, 97], random: false)
  let deconv_buf = generateORTValue(shape: [1, 64, 2, 97], random: false)
  let gridnet_bufs_buf0_K_buf = generateORTValue(shape: [4, 49, 582], random: false)
  let gridnet_bufs_buf0_V_buf = generateORTValue(shape: [4, 49, 1552], random: false)
  let gridnet_bufs_buf0_c0 = generateORTValue(shape: [1, 97, 64], random: false)
  let gridnet_bufs_buf0_h0 = generateORTValue(shape: [1, 97, 64], random: false)
  let gridnet_bufs_buf1_K_buf = generateORTValue(shape: [4, 49, 582], random: false)
  let gridnet_bufs_buf1_V_buf = generateORTValue(shape: [4, 49, 1552], random: false)
  let gridnet_bufs_buf1_c0 = generateORTValue(shape: [1, 97, 64], random: false)
  let gridnet_bufs_buf1_h0 = generateORTValue(shape: [1, 97, 64], random: false)
  let gridnet_bufs_buf2_K_buf = generateORTValue(shape: [4, 49, 582], random: false)
  let gridnet_bufs_buf2_V_buf = generateORTValue(shape: [4, 49, 1552], random: false)
  let gridnet_bufs_buf2_c0 = generateORTValue(shape: [1, 97, 64], random: false)
  let gridnet_bufs_buf2_h0 = generateORTValue(shape: [1, 97, 64], random: false)
  let istft_buf = generateORTValue(shape: [1, 2, 194, 1], random: false)
  
  let inputs = [
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
    "istft_buf": istft_buf]
  
  return inputs
}

// Function to generate ORT tensor of a given shape with the option of initializing
// with either random values or zeros.
func generateORTValue(shape: [Int], random: Bool = false) -> ORTValue {
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

  // try to create ORTValue and handle error
  let input = try! ORTValue(
    tensorData: NSMutableData(data: inputDataBytes),
    elementType: ORTTensorElementDataType.float,
    shape: inputShape)

  return input
}
