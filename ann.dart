import "dart:core";
import "dart:math";

double E = 2.7182818284;

List<List<List<double>>> input = [[[0,0,10],[0,0,1],[0,1,0],[1,0,0],[1,1,0],[1,0,1],[0,1,1],[1,1,1]],[[0, 1], [0,1], [1,0], [0,1], [1,0], [0,1], [1,0], [1,0]]];

List<int> networkShape = [3, 5, 2];
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

//List<int> networkShape = [5760, 3852, 3852, 3852, 12];
List<List<double>> inputGen = List.generate(1, (i) => (List.generate(5760, (j) => (Random().nextDouble() * 10000), growable: true)));
//List<List<List<double>>> input = [inputGen, [[1,0, 1,0, 1,0, 1,0, 1,0, 1,0]]];




void main()
{
    final stopwatch = new Stopwatch()..start();
    List<List<List<List<double>>>> networkArray = generateLayers(networkShape);
    print("Paramater initialisation completed in ${stopwatch.elapsed}\n\n");

    
    print("Forward pass:\n");
    stopwatch.reset();

    for(int i=0; i<(input[0].length); i++)
    {
        print("input: ${input[0][i]}");
        List<double> output = forwardPass((input[0][i]), networkArray, ["ReLU", "Softmax"]);
        print("Output: ${output}");
        double outputLoss = calculateLoss(output, input[1][i], "MSE");
        print("Loss: ${outputLoss}");
    }
    print("Network trained 1 epoch in: ${stopwatch.elapsed}");

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



List<double> forwardPass(List<double> inputData, List<List<List<List<double>>>> layersArray, List<String> activationFunctions)
{
    List<double> layerInput = inputData;
    List<double> outputData = [];
    for(int x=0; x<(layersArray.length); x++) //iterates thru the layers
    {
        String activationFunction = activationFunctions[x];
        List<List<List<double>>> layerArray = layersArray[x];

        List<double> layerOutput = [];
        for(int i=0; i<(layerArray[0].length); i++) //iterates thru neurons
        {

            //start function
            double neuronValue = 0.0;
            for(int j=0; j<(layerArray[0][0].length); j++) //iterates thru weights
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


List<double> softmax(List<double> layerOutput)
{
    List<double> output = [];

    double smallest = 1000000000000000;
    for(int i=0; i<(layerOutput.length); i++)
    {
        if(layerOutput[i] < smallest)
        {
            smallest = layerOutput[i];
        }
    }


    for(int i=0; i<(layerOutput.length); i++)
    {
        layerOutput[i] -= smallest;
    }


    List<double> expValues = [];
    for(int i=0; i<(layerOutput.length); i++)
    {
        expValues.add(pow(E, (layerOutput[i])).toDouble());
    }


    double normBase = 0;
    for(int i=0; i<(expValues.length); i++)
    {
        normBase += expValues[i];
    }


    for(int i=0; i<(expValues.length); i++)
    {
        output.add(expValues[i]/normBase);
    }


    return output;
}


List<double> activation(List<double> layerOutput, String activationFunction)
{

    if(activationFunction == "Softmax")
    {
        return softmax(layerOutput);
    }

    else
    {
        List<double> output = [];

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
                    y=(x*0.1);
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



double calculateLoss(List<double> output, List<double> desired, [String function = "MSE"])
{

    if(function == "MSE")
    {

        List<double> losses = [];
        double mse = 0;
        
        for(int i=0; i<(output.length); i++)
        {
            losses.add(pow((output[i]-desired[i]), 2).toDouble());
        }

        for(int i=0; i<(losses.length); i++)
        {
            mse += losses[i];
        }

        mse = mse/(losses.length);
        mse = sqrt(mse);

        return mse;
    }
    
    else
    {
        return 0.0;
    }
}


void backprop(loss, List<List<List<List<double>>>> networkArray, List<String> activationFunctions)
{
    for(int i=(networkArray.length); i>1; i--)
    {

        
    }
}
