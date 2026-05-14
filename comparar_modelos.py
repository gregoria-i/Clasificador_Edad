"""
comparar_modelos.py
script para generar el json que resume el desempeño de un modelo de redes neuronales. 
REQUIERE QUE HAYA AL MENOS 2 ARCHIVOS JSON EN LA CARPETA "resultados" PARA FUNCIONAR CORRECTAMENTE.

@author: Andrea Gregorio
@date: 2024-06
"""
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class CompararModelos:
    def __init__(self, carpeta_jsons):
        self.carpeta = carpeta_jsons
        self.modelos = []

        for archivo in os.listdir(carpeta_jsons):
            if archivo.endswith(".json"):
                path = os.path.join(carpeta_jsons, archivo)
                with open(path, "r") as f:
                    data = json.load(f)
                self.modelos.append(data)

    def resumen(self):
        """Imprime una tabla simple con el desempeño de cada modelo"""
        print("\nRESULTADOS\n")
        modelos_ordenados = sorted(self.modelos, key=lambda x: x["one_off"], reverse=True)

        for m in modelos_ordenados:
            print(f'{m["model_name"]} | test_acc={m["test_acc"]:.4f} | one_off={m["one_off"]:.4f}')

    def plot_perdida(self):
        plt.figure(figsize=(10, 5))
        for m in self.modelos:
            plt.plot(m["val_loss"], label=f'{m["model_name"]}-Validación')
            plt.plot(m["train_loss"], label=f'{m["model_name"]}-Entrenamiento')
        plt.title("PÉRDIDA")
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida")
        plt.legend()
        plt.grid()

        plt.show()

    def plot_presicion(self):
        plt.figure(figsize=(10, 5))
        for m in self.modelos:
            plt.plot(m["val_acc"], label=f'{m["model_name"]}-Validación')
            plt.plot(m["train_acc"], label=f'{m["model_name"]}-Entrenamiento')
        plt.title("PRECISIÓN")
        plt.xlabel("Épocas")
        plt.ylabel("Precisión")
        plt.legend()
        plt.grid()

        plt.show()

    def matriz_confusion(self, model_name):
        for m in self.modelos:
            if m["model_name"] == model_name:
                y_true = m["y_true"]
                y_pred = m["y_pred"]
                cm = confusion_matrix(
                    y_true,
                    y_pred
                )
                plt.figure(figsize=(8, 8))
                plt.imshow(cm)
                plt.title(model_name)
                plt.xlabel("Predicción")
                plt.ylabel("Real")
                plt.colorbar()
                plt.show()
                return
        print("Modelo no encontrado")


if __name__ == "__main__":
    carpeta_jsons = "resultados"
    comparador = CompararModelos(carpeta_jsons)
    comparador.resumen()
    comparador.plot_perdida()
    comparador.plot_presicion()
    comparador.matriz_confusion("vit_tiny_patch16_224.augreg_in21k")
    comparador.matriz_confusion("deit_tiny_patch16_224.fb_in1k")
    comparador.matriz_confusion("xcit_tiny_12_p16_224.fb_in1k")
    