import java.util.ArrayList;
import java.util.Random;
import java.util.TreeMap;

public class PerceptronMulti extends SupervisedLearner
{
    Random rand;
     PerceptronMulti(Random rand)
     {
        this.rand=rand;
     }
    Perceptron setosa;
    Perceptron versicolor;
    Perceptron virginica;

    void printLabels(int size, Matrix m)
    {
        for (int i=0; i<size; i++)
        {
            System.out.println(m.get(i,0));
        }
    }
    private void  updateLabels(Matrix features, String type, Matrix labels)
    {

    for(int i=0; i<features.rows(); i++)
    {
      if(type.equals("Iris-setosa"))
      {
          switch ((int)labels.get(i,0))
          {
              case(0):
                  labels.set(i,0,1);
                  break;
              case(1):
                  labels.set(i,0,0);
                  break;
              case(2):
                  labels.set(i,0,0);
                  break;
          }

      }
      else if(type.equals("Iris-versicolor"))
      {
          switch ((int)labels.get(i,0))
          {
              case(0):
                  labels.set(i,0,0);

                  break;
              case(1):
                  labels.set(i,0,1);

                  break;
              case(2):
                  labels.set(i,0,0);

                  break;
          }
      }
      else
      {
          switch ((int)labels.get(i,0))
          {
              case(0):
                  labels.set(i,0,0);

                  break;
              case(1):
                  labels.set(i,0,0);

                  break;
              case(2):
                  labels.set(i,0,1);

                  break;
          }
      }
    }

    }
    double []freshCopy(double[]old)
    {
        double []newone=new double[old.length];
        for(int i=0; i<old.length; i++)
        {
            newone[i]=old[i];
        }
        return newone;
    }
    ArrayList< double[] > deepCopy(ArrayList< double[] > copyier)
    {
        ArrayList< double[] >newarray=new ArrayList< double[] >();
        for(int i=0; i<copyier.size(); i++)
        {
            newarray.add(freshCopy(copyier.get(i)));
        }
        return newarray;
    }

    @Override
    public void train(Matrix features, Matrix labels) throws Exception {
         setosa=new Perceptron(rand);
         versicolor=new Perceptron(rand);
         virginica=new Perceptron(rand);
        ArrayList< double[] >originial=deepCopy(labels.m_data);
         updateLabels(features,"Iris-setosa", labels);

         setosa.train(features,labels);
          labels.m_data=originial;
          originial=deepCopy(labels.m_data);
         updateLabels(features,"Iris-virginica", labels);
        virginica.train(features,labels);
        labels.m_data=originial;
        originial=deepCopy(labels.m_data);
        updateLabels(features,"Iris-versicolor", labels);
        versicolor.train(features,labels);
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception {

        double [] setosaArr=labels.clone();
        double guess=setosa.predictSpecial(features, setosaArr);
        double [] verisocolorArr=labels.clone();
        double guess2=versicolor.predictSpecial(features,verisocolorArr);
        double[] virginicaArr=labels.clone();
        double guess3=virginica.predictSpecial(features,virginicaArr);
        //System.out.println("I TRY");
        //System.out.println(guess+ " "+guess2+" "+ guess3);
        if(guess>guess3) {
            labels[0] = 0;
        }
        else if(guess3>guess)
        {
            labels[0]=2;
        }
        else
        {
            labels[0]=1;
        }
    }
}
