public protocol TrainingResultProtocol {
    func saveLog(
        modelAuthor: String,
        modelName: String,
        modelVersion: String
    )
}
