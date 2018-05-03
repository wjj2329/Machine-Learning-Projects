import java.util.ArrayList;

public class Node {
    public ArrayList<Double>outputs=new ArrayList<>();
    public ArrayList<Node> forwardnodes=new ArrayList<>();
    public ArrayList<Node> backwadnodes=new ArrayList<>();
    public ArrayList<Node> samelevelnodes=new ArrayList<>();
    public BackProp.type type;
    public ArrayList<Double>weights=new ArrayList<>();
    public double sumationValue=0;
    public double delta=0;
    double learningrate=0.1;
    double bias=0;
    int number;
    double momentum=0;
    double previousweight=0;
    Node(BackProp.type type, double bias, int number)
    {
        this.bias=bias;
        this.type=type;
        this.number=number;
    }
    private double Sigmoid(double total)
    {
        return 1.0/(1.0+Math.pow(Math.E, (-1.0*total)));

    }
    public void updateSumationBeigginingNode( double value)
    {

        this.sumationValue=value;
    }
    public void updateSumationNormalNode()
    {
        sumationValue = 0.0;
        for(int i=0; i<backwadnodes.size(); i++)
        {
            sumationValue+=(backwadnodes.get(i).sumationValue*weights.get(i));
            //System.out.println("HI "+weights.get(i)+" "+backwadnodes.get(i).sumationValue+" to get this before sigmoid "+sumationValue+" on Node "+number);
        }
       // System.out.println("My sumation value total before bias "+sumationValue);
        sumationValue+=bias;
        sumationValue=Sigmoid(sumationValue);

    }
    public void backPropNormalNode()
    {
            double sum=0;
            for(int i=0; i<this.forwardnodes.size(); i++)
            {
              //   System.out.println("I use the delta "+forwardnodes.get(i).delta+" and i use weight "+forwardnodes.get(i).weights.get(i)+" "+number);
                sum+=(forwardnodes.get(i).delta*forwardnodes.get(i).weights.get(i));
            }
            this.delta=sum*(sumationValue*(1.0-sumationValue));
            for(int i=0; i<weights.size(); i++)
            {
                weights.set(i, weights.get(i)+delta*learningrate*weights.get(i)+momentum*previousweight);
            }
    }
    //I believe we should add the weight change that makes more sence might be subtract though.
    public void backPropEnd( double target)
    {
        delta=(target-this.sumationValue)*this.sumationValue*(1.0-sumationValue);
        for(int i=0; i<weights.size(); i++)
        {
          //System.out.println("this weight "+weights.get(i));
          weights.set(i, weights.get(i)+delta*learningrate*weights.get(i));
        }
    }

    public String nodeLoctoString()
    {
        StringBuilder dude=new StringBuilder();
        dude.append("my backward nodes are ");
        for(Node n:this.backwadnodes)
        {
            dude.append(n.number+" ");
        }
        dude.append("my same level nodes are ");
        for(Node n:this.samelevelnodes)
        {
            dude.append(n.number+" ");
        }
        dude.append("\n my forward nodes are ");
        for (Node n:this.forwardnodes)
        {
            dude.append(n.number+" ");
        }
        return dude.toString();

    }
    @Override
    public String toString()
    {
        StringBuilder dude=new StringBuilder();
        dude.append(" My type is this "+type+" my weights are this ");
        for (Double d:weights)
        {
            dude.append(d);
            dude.append(" ");
        }
        dude.append("my sigmoid on summation value is this "+sumationValue);
        dude.append(" MY Number is  "+number);
        return dude.toString();

    }

}
