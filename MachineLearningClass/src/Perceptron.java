import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class Perceptron extends SupervisedLearner {

    public Perceptron(Random rand) {
        super();
        this.myrandom=rand;
        biasweight=rand.nextDouble();//create a random biasweight with the random you give me
    }

    private double [] weights=null;
    private double bias=1;
    private Random myrandom;
    private double biasweight;
    private double learningrate=.1; //arbitrary number

    public void printArray(double [] myarray) //a helpful function for debugging.
    {
        for (int i=0; i<myarray.length; i++)
        {
            System.out.print(myarray[i]+" ");
        }
        System.out.println();

    }
    //regular training.
    @Override
    public void train(Matrix features, Matrix labels) throws Exception {
        int checkpoint=3;
        double correctP=0;
        double incorrectP=0;
        double stoppingAccuracy=.001;
        boolean stop=false;
        boolean firstTime=true;
        if(weights==null)//initalizes the weights with the random you give me.
        {
            this.weights=new double[features.cols()+1];
            for (int i=0; i<weights.length; i++) {
                weights[i]=myrandom.nextDouble();
            }
            weights[weights.length-1]=biasweight;
        }

        int epoch=0;//so we can know what cycle we are on.
        while(!stop) {
            epoch++;
            Collections.shuffle(Arrays.asList(weights));
            double correct=0;
            double incorrect=0;
            for (int i = 0; i < features.rows(); i++) {
                double yield = 0;
                for (int j = 0; j < weights.length; j++) {
                    if(j!=weights.length-1) {
                        yield += weights[j] * features.get(i, j);//update with the weights for a total aka yield
                    }else
                    {
                        yield+=weights[j];  //update with the bias
                    }
                }

                //chop off to 0 or 1 corresponding to the output/reponse
                if (yield < 0) {
                    yield = 0;
                } else if (yield > 0) {
                    yield = 1;
                }

                if (labels.get(i, 0) != yield)//if we made a mistake
                {
                    for (int x=0; x<weights.length; x++) {
                        double updateweight=0;
                        if((weights.length-1)==x)
                        {
                            if(yield==0)
                            {
                                updateweight = learningrate * bias;//update bias one way or the other
                            }
                            else
                            {
                                updateweight=learningrate*bias*-1;
                            }
                        }
                        else {
                            updateweight = learningrate * (labels.get(i, 0) - yield) * features.get(i, x);//update regular weight
                        }
                        weights[x] += updateweight;
                    }
                    incorrect++;//used below allong with the correct variable
                    }
                    else
                {
                    correct++;
                }



                }
            if(epoch%checkpoint==0)//we can now check if we can stop
            {
                if(!firstTime) {//this is how we know we can stop. see report for full description.
                    //System.out.println("correct is "+correct)
                    double prevAcc = correctP / (incorrectP + correctP);
                    double currAcc = correct / (incorrect + correct);
                    if (Math.abs(currAcc - prevAcc) <= stoppingAccuracy) {
                        stop = true;
                        System.out.println("the prev accuracy is "+prevAcc+" the current accracy is "+currAcc);
                        System.out.println("that would end being this "+Math.abs(currAcc-prevAcc));
                        System.out.println("Done with "+epoch+" ");
                    }

                }
                firstTime=false;
                correctP=correct;
                incorrectP=incorrect;

            }
        }
    }

    public double predictSpecial(double[] features, double[] labels)
    {
        double yeild=0;
        //printArray(labels);
        for (int i=0; i<features.length; i++) {//simple predict method with my weights.
            yeild+=weights[i]*features[i];
        }
        if(yeild<0)
        {
            yeild=0;
        }
        else if(yeild>1)
        {
            yeild=1;
        }
        return yeild;
    }
    @Override
    public void predict(double[] features, double[] labels) throws Exception {
        double yeild=0;
        //printArray(labels);
        for (int i=0; i<features.length; i++) {//simple predict method with my weights.
             yeild+=weights[i]*features[i];
        }
        if(yeild<0)
        {
            yeild=0;
        }
        else if(yeild>1)
        {
            yeild=1;
        }
        labels[0]=yeild;

    }
}
