import java.util.ArrayList;
import java.util.Random;

public class BackProp extends SupervisedLearner {
    Random rand;
    BackProp(Random random)
    {
        this.rand=random;
    }
    private int layers=4;
    public enum type {Start, End, Regular}
    private ArrayList<ArrayList<Node> >mylayers=null;
    private int numberofInputs=2;
    private int numberofOutputs=3;
    private int numberofNodesperHidenLayer=3;
    ArrayList<Double>testing;
    ArrayList<Double>whatIwant;
    ArrayList<Double>whatImgiven;
    double bias=0.1;
    int number=0;
    ArrayList<Double>testingWeights=new ArrayList<>();

    //Go forward through the network.  Just sum each weight with each correpsonding input and add a bias
    //Then use logisitc funciton 1/1+e^-sumfound to get true output value
    // Do the samething (using the output values from the nodes before) till you get to the end (output nodes)

    // Then Find the Error by 1/2(target-output)^2 Do the same for each output node
    //  Then sum up all of the errors you got now you have total error.
    // Now break into three parts a derivative
    //Part 1 (output-target)
    //Part 2 (output)(1-output)
    //part 3 (output of previous node)
    //mulitply the three parts of it together and subtract that number from wht weight you now have update weight
    //don;t update on the fly update at the end!
    // on next level get parts 1 and 2 muliply them  and get your weight your updating mulitply it by that.  Do same process for other thing your connected to
    //sum them up
    //then do your weight*(1-weight)
    //then do
    void printMyNetWork(ArrayList<ArrayList<Node>> mynetwork)
    {
        for(int i=0; i<mynetwork.size(); i++)
        {
            for (int j=0; j<mynetwork.get(i).size(); j++)
            {
               System.out.print(mynetwork.get(i).get(j).toString()+" ");
               //System.out.print(mynetwork.get(i).get(j).nodeLoctoString());
            }
            System.out.println();
        }
    }
    ArrayList<ArrayList<Node>> CreateMyLayers()
    {
       ArrayList<ArrayList<Node>> mysetup=new ArrayList<>();
        for(int i=0; i<layers; i++)
        {
            mysetup.add(new ArrayList<Node>());
            if(i==0) {
                for(int j=0; j<numberofInputs; j++)
                {
                    mysetup.get(i).add(new Node(type.Start, bias, ++number));
                }
            }
            else if(i==1)
            {
                for(int j=0; j<numberofNodesperHidenLayer; j++)
                {
                    mysetup.get(i).add(new Node(type.Regular, bias, ++number));
                    for(int z=0; z<numberofInputs; z++)
                    {
                        mysetup.get(i).get(j).weights.add(testingWeights.remove(0));
                    }
                }

            }
            else if(i==layers-1)
            {
                for(int j=0; j<numberofOutputs; j++)
                {
                    mysetup.get(i).add(new Node(type.End, bias,++number));
                    mysetup.get(i).get(j).outputs=whatIwant;
                }
                for(int y=0; y<numberofOutputs; y++)
                {
                    for (int h=0; h<numberofNodesperHidenLayer; h++) {
                        mysetup.get(i).get(y).weights.add(testingWeights.remove(0));
                    }
                }
            }
            else
            {
                    for (int j = 0; j < numberofNodesperHidenLayer; j++) {
                        mysetup.get(i).add(new Node(type.Regular, bias, ++number));
                    }
                    for (int y = 0; y < numberofNodesperHidenLayer; y++) {
                        for (int h = 0; h < numberofNodesperHidenLayer; h++) {

                            mysetup.get(i).get(y).weights.add(testingWeights.remove(0));
                        }
                    }
                }

        }

        System.out.println("INITIAL SET UP");
        printMyNetWork(mysetup);
        //System.exit(1);
        for(int i=0; i<mysetup.size(); i++)
        {
            for (int j=0; j<mysetup.get(i).size(); j++)
            {
                if(mysetup.get(i).get(j).type!=type.Start) {
                    for (int x = 0; x < mysetup.get(i-1).size(); x++) {
                        mysetup.get(i).get(j).backwadnodes.add(mysetup.get(i - 1).get(x));
                    }
                }

                if(mysetup.get(i).get(j).type!=type.End) {
                    for (int x = 0; x < mysetup.get(i+1).size(); x++) {
                       // System.out.println(mysetup[i][j]);
                        //System.out.println(mysetup[i+1][x]);
                        mysetup.get(i).get(j).forwardnodes.add(mysetup.get(i + 1).get(x));
                    }

                }
                for(int h=0; h<mysetup.get(i).size(); h++)
                {
                    mysetup.get(i).get(j).samelevelnodes.add(mysetup.get(i).get(h));
                }
            }

        }

        return mysetup;
    }

    void printFeatures(Matrix features)
    {
        System.out.println("FEATURES "+features.rows()+" "+features.cols());
        for(int i=0; i<features.rows(); i++)
        {
            for(int j=0; j<features.cols(); j++)
            {
                System.out.println(features.get(i,j));
            }
        }
    }
    void printLabels(Matrix labels)
    {
        System.out.println("LABELS "+labels.rows());
        for(int i=0; i<labels.rows(); i++)
        {
            System.out.println(labels.get(i,0));
        }
    }
    @Override
    public void train(Matrix features, Matrix labels) throws Exception {
         testing=new ArrayList<>();
         whatIwant=new ArrayList<>();
         whatImgiven=new ArrayList<>();
         this.mylayers=CreateMyLayers();
         System.out.println();
         printMyNetWork(mylayers);
         boolean canStop=false;
         double meansquared=0;
         double previousmeansquared=0;
         double epochcheck=5;
         double currentepoch=0;

       while(!canStop) {
             meansquared=0;
           for (int x = 0; x < features.rows(); x++) {
               whatIwant.clear();
               whatImgiven.clear();
               for (int y = 0; y < features.cols(); y++) {
                   whatImgiven.add(features.get(x, y));
               }
               whatIwant.add(0.1);
               whatIwant.add(1.0);
               //for (int p = 0; p < this.numberofInputs; p++) {
               //  whatIwant.add(0.0);
               //}
               //whatIwant.set((int) labels.get(x, 0), 1.0);
               for (Double d : whatIwant) {
                   System.out.println(d);
               }
               for (Double d : whatImgiven) {
                   System.out.println(d);
               }

               for (int i = 0; i < this.mylayers.size(); i++) {
                   for (int j = 0; j < this.mylayers.get(i).size(); j++) {
                       if (mylayers.get(i).get(j).type == type.Regular || mylayers.get(i).get(j).type == type.End) {
                           mylayers.get(i).get(j).updateSumationNormalNode();
                       } else if (mylayers.get(i).get(j).type == type.Start) {
                           mylayers.get(i).get(j).updateSumationBeigginingNode(whatImgiven.get(j));
                       }
                   }

               }

               System.out.println("after forward pass through ");
               printMyNetWork(mylayers);
               //System.exit(0);

               for (int i = mylayers.size() - 1; i >= 0; i--) {
                   for (int j = 0; j < mylayers.get(i).size(); j++) {
                       if (mylayers.get(i).get(j).type == type.Start) {
                           continue;
                       }
                       if (mylayers.get(i).get(j).type == type.End) {
                           mylayers.get(i).get(j).backPropEnd(whatIwant.get(j));
                           meansquared+=(mylayers.get(i).get(j).sumationValue*mylayers.get(i).get(j).sumationValue);
                       }
                       if (mylayers.get(i).get(j).type == type.Regular) {
                           mylayers.get(i).get(j).backPropNormalNode();
                       }
                   }
               }
               System.out.println("FINAL RESULT");
               printMyNetWork(this.mylayers);
               meansquared=Math.sqrt(meansquared);
               currentepoch++;
               if(currentepoch%epochcheck==0)
               {
                   if(meansquared>previousmeansquared+0.5)
                   {
                       return;
                   }
                   previousmeansquared=meansquared;
               }
           }


       }

      // }


    }


//need to see the error rate  Mean squared error.
    @Override
    public void predict(double[] features, double[] labels) throws Exception {

        //System.out.println("I predict ");
        ArrayList<Double>inputs=new ArrayList<>();
        for(int i=0; i<features.length; i++)
        {
           // System.out.println(features[i]);
            inputs.add(features[i]);
        }
        for (int i = 0; i < mylayers.size(); i++) {
            for (int j = 0; j < mylayers.get(i).size(); j++) {
                if (mylayers.get(i).get(j).type == type.Regular || mylayers.get(i).get(j).type == type.End) {
                    mylayers.get(i).get(j).updateSumationNormalNode();
                } else if (mylayers.get(i).get(j).type == type.Start) {
                    mylayers.get(i).get(j).updateSumationBeigginingNode(inputs.get(j));
                }
            }

        }
        double value=mylayers.get(mylayers.size()-1).get(0).sumationValue;
        double value2=mylayers.get(mylayers.size()-1).get(1).sumationValue;
          double value3=mylayers.get(mylayers.size()-1).get(2).sumationValue;
          if(value>value2&&value>value3)
        {
            labels[0]=0;
        }
        else if(value2>value&&value3>value)
        {
            labels[0]=1;
        }
        else {
            labels[0]=2;
        }

    }

}
