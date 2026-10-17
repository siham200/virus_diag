class ClinicalPredictor:
    """
    Classe qui reçoit un modèle déjà entraîné et fait le diagnostic.
    """

    def __init__(self, model):
        """
        :param model: Un modèle IA déjà entraîné disposant d'une méthode predict()
        """
        self.model = model

    def diagnose(self, patient_data):
        """
        :param patient_data: Liste ou tableau de données du patient
                             [age, temperature, toux, fatigue, mal_de_gorge, congestion]
        :return: "Infecté" si prédiction >= 0.5, "Sain" sinon
        """
        prediction = self.model.predict([patient_data])[0]

        if prediction >= 0.5:
            return "Infecté"
        else:
            return "Sain"


# --- Test de la classe ---
if __name__ == "__main__":
    from core.logistic_regression import LogisticRegressionModel
    from pipeline.trainer import Trainer

    # 1. Entraîner le modèle
    model = LogisticRegressionModel()
    trainer = Trainer(model)
    trainer.train()

    # 2. Créer le ClinicalPredictor avec le modèle entraîné
    predictor = ClinicalPredictor(model)

    # 3. Tester avec un patient
    # Patient : age=45, temp=38.5, toux=1, fatigue=1, mal_de_gorge=1, congestion=1
    patient = [45, 38.5, 1, 1, 1, 1]
    resultat = predictor.diagnose(patient)
    print(f"🩺 Diagnostic : {resultat}")

    # Patient sain
    patient_sain = [25, 37.2, 0, 1, 0, 0]
    resultat2 = predictor.diagnose(patient_sain)
    print(f"🩺 Diagnostic : {resultat2}")