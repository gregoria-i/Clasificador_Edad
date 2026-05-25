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
import pandas as pd


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
        modelos_ordenados = sorted(self.modelos, key=lambda x: x["one_off"], reverse=True)

        datos = []

        for m in modelos_ordenados:
            datos.append({"Modelo": m["model_name"], "Test Accuracy": m["test_acc"],"One-Off": m["one_off"]})

        df = pd.DataFrame(datos)
        print("RESULTADOS\n")
        print(df.to_string(index=False))

    def plot_perdida(self, modelos_requeridos):
        modelos = []
        for nombre in modelos_requeridos:
            for m in self.modelos:
                if m["model_name"] == nombre:
                    modelos.append(m)
                    break
        fig, axs = plt.subplots(2,3,figsize=(15, 8))
        axs = axs.flatten()
        for ax, m in zip(axs, modelos):
            ax.plot(m["train_loss"], label="Entrenamiento")
            ax.plot(m["val_loss"], label="Validación")
            ax.set_title(m["model_name"])
            ax.set_xlabel("Épocas")
            ax.set_ylabel("Pérdida")
            ax.grid()

        for i in range(len(modelos), len(axs)):
            axs[i].axis("off")

        handles, labels = axs[0].get_legend_handles_labels()

        fig.legend(handles, labels, loc="upper right", ncol=2)
        fig.suptitle("Comparación de pérdida", fontsize=16)
        plt.tight_layout(rect=[0,0,1,0.95])

        plt.show()

    def plot_precision(self, modelos_requeridos):
        modelos = []
        for nombre in modelos_requeridos:
            for m in self.modelos:
                if m["model_name"] == nombre:
                    modelos.append(m)
                    break

        fig, axs = plt.subplots(2,3,figsize=(15, 8))
        axs = axs.flatten()
        for ax, m in zip(axs, modelos):
            ax.plot(m["train_acc"], label="Entrenamiento")
            ax.plot(m["val_acc"], label="Validación")
            ax.set_title(m["model_name"])
            ax.set_xlabel("Épocas")
            ax.set_ylabel("Precisión")
            ax.grid()

        for i in range(len(modelos), len(axs)):
            axs[i].axis("off")

        handles, labels = axs[0].get_legend_handles_labels()

        fig.legend(handles, labels, loc="upper right", ncol=2)
        fig.suptitle("Comparación de precisión", fontsize=16)
        plt.tight_layout(rect=[0,0,1,0.95])

        plt.show()

        
    def matriz_confusion(self, modelos_requeridos):
        modelos = []
        for nombre in modelos_requeridos:
            for m in self.modelos:
                if m["model_name"] == nombre:
                    modelos.append(m)
                    break

        fig, axs = plt.subplots(2,3,figsize=(15, 8))
        axs = axs.flatten()
        for ax, m in zip(axs, modelos):
            cm = confusion_matrix(
                m["y_true"],
                m["y_pred"]
            )

            im = ax.imshow(cm)
            ax.set_title(m["model_name"])
            ax.set_xlabel("Predicción")
            ax.set_ylabel("Real")

        for i in range(len(modelos), len(axs)):
            axs[i].axis("off")

        fig.subplots_adjust(right=1)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        fig.suptitle("Matrices de confusión", fontsize=16)
        plt.tight_layout(rect=[0,0,1,0.95])

        plt.show()


if __name__ == "__main__":
    carpeta_jsons = os.path.join("dev_model", "resultados")
    comparador = CompararModelos(carpeta_jsons)
    # Imprime la tabla general ordenada de mayor one-off a menor
    comparador.resumen()
    
    # Modelos pasados
    deit_tiny = "deit_tiny_patch16_224.fb_in1k"
    xcit_tiny = "xcit_tiny_12_p16_224.fb_in1k"
    vit_tiny = "vit_tiny_patch16_224.augreg_in21k"

    # Modelos sugeridos
    deit3_small = "deit3_small_patch16_224.fb_in22k_ft_in1k"
    xcit_small = "xcit_small_12_p16_224.fb_in1k"
    swin_tiny = "swin_tiny_patch4_window7_224.ms_in1k"

    comparador.plot_perdida([deit_tiny, xcit_tiny, vit_tiny, deit3_small, xcit_small, swin_tiny])
    comparador.plot_precision([deit_tiny, xcit_tiny, vit_tiny, deit3_small, xcit_small, swin_tiny])
    comparador.matriz_confusion([deit_tiny, xcit_tiny, vit_tiny, deit3_small, xcit_small, swin_tiny])
