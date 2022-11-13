import "dart:core";
import "dart:math";


void main()
{
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


    List<List<List<List<double>>>> networkArray = generateLayers(networkShape);
    print("\n\n\n ${networkArray}");

    //print("\n\n\nForward pass:");
    //print(forwardPass(forwardPass(input[0], layer1), layer2));
    print("\n\n\nForward pass 2");
    print("FP2 output: ");
    print(forwardPass2(input[0], networkArray));
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



List<double> forwardPass(inputData, layersArray)
{
    List<double> layerInput = inputData;
    List<double> outputData = [];
    print(layersArray.length);
    for(int x=0; x<(layersArray.length)-0; x++)
    {
        print("Layer ${x+1} input: ${layerInput}");
        List<List<List<double>>> layerArray = layersArray[x];

        List<double> layerOutput = [];
        for(int i=0; i<(layerArray[0].length)-0; i++)
        {

            //start function
            double neuronValue = 0.0;
            for(int j=0; j<(layerArray[0][0].length)-1; j++)
            {
                neuronValue += (layerInput[j].toDouble()) * (layerArray[0][i][j].toDouble());

            }
            neuronValue += layerArray[1][i][0];
            //end function

            layerOutput.add(neuronValue);
        }
        
        layerInput = layerOutput;
        print("layerOutput ${layerOutput}");
        outputData = layerOutput;
    }
    return outputData;
}



