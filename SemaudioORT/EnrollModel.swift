// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import Foundation
import onnxruntime_objc

// NOTE: Files are saved at:
// /Users/yourusername/Library/Developer/CoreSimulator/Devices/DEVICE_ID/data/Containers/Data/Application/APPLICATION_ID/Documents

class EnrollModel {
    private let ortEnv: ORTEnv
    private let ortSession: ORTSession
    
    enum ModelError: Error {
        case Error(_ message: String)
    }
    
    init() throws {
        ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        guard let modelPath = Bundle.main.path(forResource: "enroll", ofType: "ort") else {
            throw ModelError.Error("Failed to find model file.")
        }
        ortSession = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: nil)
    }
    
    // Returns 1d array of zeros or random vals
    func generateArray(size: Int, random: Bool = true) -> [Float] {
        // random array of floats
        let Data: [Float]
        if random {
            Data = (0..<size) .map { _ in Float.random(in: -1...1) }
        } else {
            Data = [Float](repeating: 0.0, count: size)
        }
        
        return Data
    }
    
    // Converts a given float array to an ortvalue of given shape
    func ArrayToORTVal(inputData: [Float], shape: [Int]) -> ORTValue {
        // Convert inputData to Bytes
        let inputDataBytes = Data (
            bytes: inputData,
            count: inputData.count * MemoryLayout<Float>.stride
        )
        
        let inputShape: [NSNumber] = shape.map { NSNumber(value: $0) }
        
        // convert to ORT value of given shape
        let value = try! ORTValue(
            tensorData: NSMutableData(data: inputDataBytes),
            elementType: ORTTensorElementDataType.float,
            shape: inputShape)
        
        return value
    }
    
    // Saves the given array at the project directory as a json
    func saveArray(Data: [Float], filename: String) {
        let documentsDirectoryURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileURL = documentsDirectoryURL.appendingPathComponent("\(filename).json")
        
        if let jsonData = try? JSONSerialization.data(withJSONObject: Data, options: []) {
            do {
                // Save the tensor as a json.
                try jsonData.write(to: fileURL)
                print("Saved \(filename) to disk!")
                return
            } catch {
                print("Error writing JSON file: \(error)")
                return
            }
        }
        print("couldn't deserialize \(filename) on write")
        return
    }
    
    // Loads the given array from the project directory
    func loadArray(filename: String) -> [Float] {
        var readData: [Float] = []
        guard let documentsDirectoryURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("Error: Could not find the Documents directory.")
            return []
        }
        let fileURL = documentsDirectoryURL.appendingPathComponent("\(filename).json")

        do {
            let data = try Data(contentsOf: fileURL)
            let jsonObject = try JSONSerialization.jsonObject(with: data, options: [])
            
            // Deserialize the JSON data back into an array of Floats
            if let jsonArray = jsonObject as? [Double] {
                readData = jsonArray.map { Float($0) }
                return readData
            } else {
                print("Error: Could not deserialize \(filename) into a [Float]")
            }
        } catch {
            print("Error reading JSON file: \(error)")
        }
        return readData
    }
    
    func runEnroll(inputORT: ORTValue) -> [Float] {
        if let outputORT = try! ortSession.run(
            withInputs: ["enrollment": inputORT],
            outputNames: ["embeding"],
            runOptions: nil)["embeding"] {
            let tensorData = try! outputORT.tensorData() as Data
            
            let floatArray = tensorData.withUnsafeBytes { buffer -> [Float] in
                let floatBuffer = buffer.bindMemory(to: Float.self)
                return Array(floatBuffer)
            }
            return floatArray
        }
        return []
    }

    func currentTimestamp() -> String {
        let now = Date()
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMddHHmmss" // Year, month, day, hour, minute, second
        return formatter.string(from: now)
    }
    
    func eval() {
        let enrollInArray = generateArray(size: 2 * 80000)
        let enrollIn = ArrayToORTVal(inputData: enrollInArray, shape: [1, 2, 80000])
        let enrolloutArray = runEnroll(inputORT: enrollIn)
        let timestamp = currentTimestamp()
        
        // Save the input and output of the model as 1d arrays in json files
        saveArray(Data: enrollInArray, filename: "enrollInArray\(timestamp)")
        saveArray(Data: enrolloutArray, filename: "enrolloutArray\(timestamp)")
    }
}
