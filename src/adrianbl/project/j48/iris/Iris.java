/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package adrianbl.project.j48.iris;

// Importación de clases requeridas
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

/**
 *
 * @author AdrianBL
 */

public class Iris {
    // Método del controlador principal
    public static void main(String args[])
    {
 
        // Try Catch para Intentar bloquear para buscar excepciones
        try {
 
            // Creando clasificador J48
            J48 j48Classifier = new J48();
 
            // Ruta del conjunto de datos
            String irisDataset = "C:\\Program Files\\Weka-3-8-6\\data\\iris.arff";
 
            // Crear bufferedreader para leer el conjunto de datos
            BufferedReader bufferedReader = new BufferedReader(new FileReader(irisDataset));
 
            // Crear instancias de conjuntos de datos
            Instances datasetInstances = new Instances(bufferedReader);
 
            // Establecer clase de destino
            datasetInstances.setClassIndex(datasetInstances.numAttributes() - 1);
 
            // Evaluation - Evaluation : OBJ
            Evaluation evaluation = new Evaluation(datasetInstances);
 
            // Cross Validate Model con 10 folds
            evaluation.crossValidateModel(
                j48Classifier, datasetInstances, 10,
                new Random(1));
            
            //Sumarry Resultados
            System.out.println(evaluation.toSummaryString("\nResultados", false));
            
            //Detailed Accuracy By Class - Resultados de los detalles de las Clases
            System.out.println(evaluation.toClassDetailsString());
            
            //Matriz de confusión
            evaluation.confusionMatrix();
            
            //Resultados
            System.out.println(evaluation.toMatrixString());
            
            
            // Árbol de decisión
            String[] options = new String[]{};
            j48Classifier.setOptions(options);
            j48Classifier.buildClassifier(datasetInstances);
            
            // Árbol de decisiones de salida - Resultados
            System.out.println("Modelo de arbol de decision:\n"+j48Classifier);
            
            // Código fuente de salida que implementa el árbol de decisión
            System.out.println("Codigo fuente:\n"+j48Classifier.toSource("ActivityRecognitionEngine"));
	
            /*
            // Verifique la precisión del modelo utilizando 10-fold cross-validation
            Evaluation eval = new Evaluation(datasetInstances);
            eval.crossValidateModel(j48Classifier, datasetInstances, 10, new Random(1), new String[] {});
            System.out.println("Model performance:\n"+eval.toSummaryString());
            */
        }
 
        // Capturar el catch para comprobar si hay excepciones
        catch (Exception e) {
 
            // Imprimir y mostrar el mensaje de la pantalla
            // usando el método getMessage()
            System.out.println("Se produjo un error!!! \n" + e.getMessage());
        }
 
        // Mostrar mensaje para imprimir en la consola
        // cuando el programa se ejecuta correctamente
        System.out.println("Ejecutado con exito.");
    }
}
