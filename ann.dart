import "dart:core";
import "dart:math";

double E = 2.7182818284;

List<List<List<double>>> input = [[[0,0,0],[0,0,1],[0,1,0],[1,0,0],[1,1,0],[1,0,1],[0,1,1],[1,1,1]],[[0,1], [0], [1], [0], [1], [0], [1], [1]]];

List<int> networkShape = [3, 3, 2];
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
//List<List<double>> inputGen = List.generate(1, (i) => (List.generate(5760, (j) => (Random().nextDouble() * 10000), growable: true)));
//List<List<List<double>>> input = [inputGen, [[1,0, 1,0, 1,0, 1,0, 1,0, 1,0]]];




void main()
{
    final stopwatch = new Stopwatch()..start();
    List<List<List<List<double>>>> networkArray = generateLayers(networkShape);
    print("Paramater initialisation completed in ${stopwatch.elapsed}\n\n");
    print(networkArray);
    print("Forward pass:\n");

    stopwatch.reset();

    for(int i=0; i<(/*input[0].length*/1); i++)
    {
        print("input: ${input[0][i]}");
        List<double> output = forwardPass((input[0][i]), networkArray, ["ReLU", "ReLU", "ReLU", "ReLU"]);
        print("Output: ${output}");
        List<double> outputLoss = calculateLoss(output, input[1][i], "MSE");
        print("Loss: ${outputLoss}\n");

        backprop(output, outputLoss, networkArray, ["ReLU", "ReLU", "ReLU", "ReLU"]);

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

        layerOutput = activation(layerOutput, activationFunction);
        layerInput = layerOutput;
        outputData = layerOutput;
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



List<double> calculateLoss(List<double> output, List<double> desired, [String function = "MSE"])
{

    if(function == "MSE")
    {
    
        List<double> losses = [];
        double sum = 0;
        
        for(int i=0; i<(output.length); i++)
        {
            losses.add(pow((desired[i] - output[i]), 2).toDouble()/2);
        }

        for(int i=0; i<(losses.length); i++)
        {
           sum += losses[i]; 
        }
        
        losses.insert(0, sum);

        return losses;
    }

    
    else
    {
        return [0];
    }

}


void backprop(List<double> networkOutputs, List<double> loss, List<List<List<List<double>>>> networkArray, List<String> activationFunctions)
{
    print("test"); 
    for(int i=(networkArray.length)-1; i>=0; i--) //iterates thru layers
    {
        String activationFunction = activationFunctions[i];
        print("1");
        print(networkArray[i]);

        for(int j=(networkArray[i][0][0].length)-1; j>=0; j--) //iterates thru neurons in weights array
        {

            for(int k=(networkArray[i][0].length)-1; k>=0; k--) //iterates thru weights
            {
                print(networkArray[i][0][k][j]); 
            }

        } 
    }
}
