package hatoark;
import java.util.ArrayList;

public class NN3layers {
	
	//
	private double influ_weight_coeff = -0.2;
	
	//入力ノード・中間ノード・出力ノードの数
	private int input_length;
	private int center_length;
	private int output_length;
	
	//入力ノード・中間ノード・出力ノードの値格納用リスト
	private ArrayList<Double> input_layer = new ArrayList<Double>();
	private ArrayList<Double> center_layer = new ArrayList<Double>();
	private ArrayList<Double> output_layer = new ArrayList<Double>();
	
	//入力層と中間層、中間層と出力層の重み格納用リスト
	//
	private ArrayList<ArrayList> input_center_weight = new ArrayList<ArrayList>();
	private ArrayList<ArrayList> center_output_weight = new ArrayList<ArrayList>();
	
	//誤差格納用リスト
	private ArrayList<Double> gosa_output;
	private ArrayList<Double> gosa_center;
	
	
	//コンストラクタ
	//インプットサイズ,中間層サイズ,アウトプットサイズを指定
	NN3layers(int input_size, int center_size, int output_size){
		
		//入力ノード・中間ノード・出力ノードの数
		this.input_length = input_size;
		this.center_length = center_size;
		this.output_length = output_size;
		
		//入力・中間・出力層のインスタンスを生成
		input_layer = new ArrayList<Double>(this.input_length + 1);
		center_layer = new ArrayList<Double>(this.center_length + 1);
		output_layer = new ArrayList<Double>(this.output_length);
		//入力・中間層の初期化(by 0)
		initializeInputCenterNode(this.input_layer, this.input_length);
		initializeInputCenterNode(this.center_layer, this.center_length);
		initializeOutputNode(this.output_layer, this.output_length);
		
		//誤差格納用リストをインスタンス
		gosa_output = new ArrayList<Double>(this.output_length);
		gosa_center = new ArrayList<Double>(this.center_length);
		//誤差リストの初期化(by 0)
		initializeGosa(this.gosa_output, this.output_length);
		initializeGosa(this.gosa_center, this.center_length);
		
		//入力層と中間層、中間層と出力層の重みの初期化(by 0.0~1.0)
		initializeWeight(this.input_center_weight, this.input_length, this.center_length);
		initializeWeight(this.center_output_weight, this.center_length, this.output_length);
	}
	
	//リストを0で初期化(ただし、末尾に1を"追加")
	private void initializeInputCenterNode(ArrayList<Double> list, int list_length){
		for(int i=0; i<list_length; i++){
			list.add(0.0);
		}
		//入力・中間層は一番最後に1が入る（length+1）
		list.add(1.0);
	}
	
	//リストを0で初期化
	private void initializeOutputNode(ArrayList<Double> list, int list_length){
		for(int i=0; i<list_length; i++){
			list.add(0.0);
		}
	}
	
	//誤差格納用リストを初期化(by 0.0)
	private void initializeGosa(ArrayList<Double> gosa_list, int list_length){
		if(!gosa_list.isEmpty()) gosa_list.clear();
		
		for(int i=0; i<list_length; i++){
			gosa_list.add(0.0);
		}
	}
	
	//二次元リストをランダム数(0.0<=1.0)で初期化
	private void initializeWeight(ArrayList<ArrayList> list, int row_size, int col_size){
		
		//rowは上層（上層は1つノードが多いことに注意）
		for(int i=0; i<row_size+1; i++){
			
			ArrayList<Double> cols = new ArrayList<Double>();
			for(int j=0; j<col_size; j++){
				cols.add(Math.random());
			}
			list.add(cols);
		}
	}
	
	//1周だけ学習を行う
	public void study(ArrayList<Integer> input_list, ArrayList<Integer> teach_list){
		
		//****入力->出力方向
		//入力にセット
		setInputLayer(input_list);		
		//中間ノードの更新
		setCenterLayer();		
		//出力ノードの更新
		setOutputLayer();
		
		//****誤差伝播
		//出力誤差を計算
		updateGosaOutput(teach_list);
		//中間層誤差を計算
		updateGosaCenter();
		//中間層と出力層間の重みを更新する
		updateCenterOutputWeight();
		//入力層と中間層間の重みを更新する
		updateInputCenterWeight();
		
	}
	
	//入力層に値を格納
	private void setInputLayer(ArrayList<Integer> input_list){
		for(int i=0; i<input_length; i++){
			input_layer.set(i, (double)input_list.get(i));
		}
	}
	
	//中間層に値を格納
	private void setCenterLayer(){
		for(int i=0; i<center_length; i++){
			for(int j=0; j<input_length+1; j++){

				center_layer.set(i, center_layer.get(i) + input_layer.get(j) * 
								(double)(input_center_weight.get(j).get(i)));
			}
			center_layer.set(i, sigmoid(center_layer.get(i)));
		}
	}
	
	//出力層に値を格納
	private void setOutputLayer(){
		for(int i=0; i<output_length; i++){
			for(int j=0; j<center_length+1; j++){
				output_layer.set(i, output_layer.get(i) + center_layer.get(j) * (double)(center_output_weight.get(j).get(i)));
			}
			output_layer.set(i, sigmoid(output_layer.get(i)));
		}
	}
	
	//出力層の誤差の値を格納
	private void updateGosaOutput(ArrayList<Integer> teach_list){
		for(int i=0; i<output_length; i++){
			gosa_output.set(i, -1.0 * (teach_list.get(i) - output_layer.get(i)) * output_layer.get(i) * (1.0 - output_layer.get(i)));
		}
	}
	
	//中間層の誤差を格納
	private void updateGosaCenter(){
		initializeGosa(gosa_center, gosa_center.size());
		
		for(int i=0; i<center_length; i++){
			for(int j=0; j<output_length; j++){
				
				gosa_center.set(i, gosa_center.get(i) + ((double)(center_output_weight.get(i).get(j)) * gosa_output.get(j)));
			}
		}
	}
	
	//中間層と出力層間の重みを更新する
	private void updateCenterOutputWeight(){
		for(int i=0; i<output_length; i++){
			for(int j=0; j<center_length+1; j++){
				
				center_output_weight.get(j).set(i, (double)(center_output_weight.get(j).get(i)) 
								+ (influ_weight_coeff * gosa_output.get(i) * center_layer.get(j)));
			}
		}
	}
	
	//入力層と中間層間の重みを更新する
	private void updateInputCenterWeight(){
		for(int i=0; i<center_length; i++){
			for(int j=0; j<input_length+1; j++){
				input_center_weight.get(j).set(i, (double)(input_center_weight.get(j).get(i)) 
								+ (influ_weight_coeff * gosa_center.get(i) * input_layer.get(j)));
			}
		}
	}
	
	
	//予想結果を返す
	public ArrayList<Double> getResult(ArrayList<Integer> input){
		//入力層に値をセット
		setInputLayer(input);
		//中間ノードの更新
		setCenterLayer();		
		//出力ノードの更新
		setOutputLayer();
		//結果（出力層）を返却
		return output_layer;
	}
	
	
	//シグモイド関数
	private double sigmoid(double val){
		return 1 / (1 + Math.exp(-1.0 * val));
	}
	
	//重み更新時の係数を変更
	public void setInfluWeightCoeff(double val){
		this.influ_weight_coeff = val;
	}
	
}
