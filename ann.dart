import "dart:core";
import "dart:math";


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


void main()
{
    List<List<List<List<double>>>> networkArray = generateLayers(networkShape);
    print("\n\n\n ${networkArray}");

    print("\n\n\nForward pass");
    print(forwardPass(input[0], networkArray, "Linear"));
    print(forwardPass(input[0], networkArray, "ReLU"));
}



List<List<List<List<double>>>> generateLayers(shape)
{
    List<List<List<List<double>>>> networkArray = [];

    for(int x=0; x<(shape.length)-1; x++)
    {
        networkArray.add([(List.generate(shape[x+1], (i) => List.generate(shape[x], (j) => Random().nextDouble(), growable: false))), (List.generate(shape[x+1], (i) => List.generate(1, (j) => Random().nextDouble(), growable:false)))]);
    }

    return networkArray;
}



List<double> forwardPass(inputData, layersArray, activationFunction)
{
    List<double> layerInput = inputData;
    List<double> outputData = [];
    print(layersArray.length);
    for(int x=0; x<(layersArray.length); x++)
    {
        print("Layer ${x+1} input: ${layerInput}");
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
    print(activationFunction);
    return outputData;
}


List<double> activation(layerOutput, activationFunction)
{
    List<double> output = [];

    for(int i=0; i<(layerOutput.length); i++)
    {
        double y = 0;
        double x = layerOutput[i];

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

        if(activationFunction == "Sigmoid")
        {
            y=x;
        }
        
        output.add(y);
    }

    return output;

}
