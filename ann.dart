import "dart:core";
import "dart:math";

double E = 2.7182818284;

List<List<double>> input = [[0,0,0],[0,0,1],[0,1,0],[1,0,0],[1,1,0],[1,0,1],[0,1,1],[1,1,1]];
List<int> networkShape = [3, 5, 1];
List<List<List<double>>> layer1ShapeExample = [ 
                                                  [ 
                                                      [1,1,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1] 
                                                  ],

                                                  [
                                                      [0], [0], [0], [0], [1]
                                                  ]
                                              ];
int batchSize = 8;
int epochs = 10;

//List<int> networkShape2 = [5760, 3852, 3852, 3852, 12];
//List<List<double>> input2 = List.generate(10, (i) => List.generate(5760, (j) => Random().nextDouble(), growable: false));



void main()
{
    final stopwatch = new Stopwatch()..start();


    List<List<List<List<double>>>> networkArray = generateLayers(networkShape);
    //print("\n\n\n ${networkArray}");
    print("Completed in ${stopwatch.elapsed}");

    
    print("\n\n\nForward pass");
    stopwatch.reset();
    //forwardPass(input2[0], networkArray, ["ReLU", "ReLU", "ReLU", "ReLU"]);
    print(forwardPass(input[6], networkArray, ["Sigmoid", "Softmax"]));
    print("Completed in ${stopwatch.elapsed}");

    print(activation([0.1, 0.9, 14, 6, 2], "Softmax"));
}



List<List<List<List<double>>>> generateLayers(shape)
{
    List<List<List<List<double>>>> networkArray = [];

    for(int x=0; x<(shape.length)-1; x++)
    {
        networkArray.add([(List.generate(shape[x+1], (i) => List.generate(shape[x], (j) => Random().nextDouble(), growable: false))), (List.generate(shape[x+1], (i) => List.generate(1, (j) => (Random().nextInt(2000000000)-1000000000)/1000000000, growable:false)))]);
    }

    return networkArray;
}



List<double> forwardPass(inputData, layersArray, List<String> activationFunctions)
{
    List<double> layerInput = inputData;
    List<double> outputData = [];
    for(int x=0; x<(layersArray.length); x++)
    {
        String activationFunction = activationFunctions[x];
        List<List<List<double>>> layerArray = layersArray[x];

        List<double> layerOutput = [];
        for(int i=0; i<(layerArray[0].length); i++)
        {

            //start function
            double neuronValue = 0.0;
            for(int j=0; j<(layerArray[0][0].length); j++)
            {
                neuronValue += (layerInput[j].toDouble()) * (layerArray[0][i][j].toDouble());

            }
            neuronValue += layerArray[1][i][0];
            //end function

            layerOutput.add(neuronValue);
        }
        
        layerInput = activation(layerOutput, activationFunction);
        outputData = activation(layerOutput, activationFunction);
    }
    return outputData;
}


List<double> activation(layerOutput, activationFunction)
{
    List<double> output = [];


    if(activationFunction == "Softmax")
    {
        double smallest = 1000000000000000;
        for(int i=0; i<(layerOutput.length); i++)
        {
            if(layerOutput[i] < smallest)
            {
                smallest = layerOutput[i];
            }

        }


        List<double> expValues = [];
        for(int i=0; i<(layerOutput.length); i++)
        {
            expValues.add((pow(E, layerOutput[i]).toDouble())-smallest);
        }

        double normBase = 0;
        for(int i=0; i<(expValues.length)-1; i++)
        {
            normBase += expValues[i];
        }

        for(int i=0; i<(expValues.length); i++)
        {
            output.add(expValues[i]/normBase);
        }
    
        return output;
    }

    else
    {
    
        for(int i=0; i<(layerOutput.length); i++)
        {
            double y = 0;
            double x = (layerOutput[i]).toDouble();

            if(activationFunction == "ReLU")
            {
                if(x<0)
                {
                    y=0;
                }


                if(x>=0)
                {
                    y=x;
                }
            }


            if(activationFunction == "LReLU")
            {
                if(x<0)
                {
                    y=(x*0.2);
                }

                if(x>=0)
                {
                    y=x;
                }
            }

            if(activationFunction == "Linear")
            {
                y=x;
            }

            output.add(y);
        }

        return output;
    }

}
