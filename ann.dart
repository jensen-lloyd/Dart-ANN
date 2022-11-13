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


        List<List<List<double>>> layer2ShapeExample = [
                                          [
                                              [1,1,1], [1,1,1], [1,1,1], [1,1,1], [1,1,1]
                                          ],

                                          [
                                              [1], [0], [0], [0], [1]
                                          ]
                                      ];


    List<List<List<double>>> layer1 = [ (List.generate(5, (i) => List.generate(3, (j) => Random().nextDouble(), growable: false))), (List.generate(5, (i) => List.generate(1, (j) => Random().nextDouble(), growable: false))) ];
    //print(layer1);

    List<List<List<double>>> layer2 = [ (List.generate(1, (i) => List.generate(5, (j) => Random().nextDouble(), growable: false))), (List.generate(1, (i) => List.generate(1, (j) => Random().nextDouble(), growable:false))) ];
    //print("\n\n ${layer2}");


    List<List<List<List<double>>>> networkArray = generateLayers(networkShape);
    print(layer1);
    print("\n ${layer2}");
    
    print("\n\n\n ${networkArray}");

    print("\n\n\nForward pass:");
    print(forwardPass(forwardPass(input[0], layer1), layer2));
    print("\n\n\nForward pass 2:");
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



List<double> forwardPass(inputData, layerArray)
{
    List<double> outputData = [];
    for(int i=0; i<(layerArray[0].length)-0; i++)
    {

        //start function
        double neuronValue = 0.0;
        for(int j=0; j<(layerArray[0][0].length)-0; j++)
        {
            neuronValue += (inputData[j].toDouble()) * (layerArray[0][i][j].toDouble()); 

        }
        neuronValue += layerArray[1][i][0];
        //end function

        outputData.add(neuronValue);
    }

    return outputData;
}


List<double> forwardPass2(inputData, layersArray)
{
    List<double> layerInput = inputData;
    List<double> outputData = [];
    for(int x=0; x<(layersArray.length)-1; x++)
    {
        List<List<List<double>>> layerArray = layersArray[x];

        List<double> layerOutput = [];
        for(int i=0; i<(layerArray[0].length)-1; i++)
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
        outputData = layerOutput;
    }
    return outputData;
}


