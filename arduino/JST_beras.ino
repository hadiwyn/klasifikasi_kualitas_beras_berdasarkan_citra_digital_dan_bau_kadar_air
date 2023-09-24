#define S0 A0   //sensor gas1
#define S1 A1   //sensor gas2
#define S2 A2   //sensor gas3
#define S3 A4   //sensor gas4
#define S4 A5   //sensor moisture

const int InputNodes = 4;
const int HiddenNodes = 10;

const float _w1 [HiddenNodes][InputNodes] = {
  {-1.4350, 0.3323, -1.3734, 1.4524},
  {-1.6556, 0.3304, -1.7725, 0.7033},
  {1.1138, 1.2265, -0.1956, 1.5903},
  {1.5901, -1.5780, 0.48650, -1.7586},
  {0.7130, 1.5987, 2.5658, -0.4220},
  {-0.2492, 1.3953, 2.6863, 0.1883},
  {1.4575, -0.7112, 1.6588, -0.9636},
  {-0.3701, -1.8959, 1.0071, 1.1149},
  {-0.7891, 1.9513, 1.6162, 0.8991},
  {-0.3889, 1.6136, 0.1635, 1.1076}
};

const float _b1 [] = {
  2.5124,
  1.8388,
  -1.5447,
  -1.6769,
  -0.7331,
  -0.5289,
  0.7596,
  -1.6839,
  -1.5249,
  -2.9571
};

const float _w2u [] = {0.0703, -0.5898, -0.2357, -1.8122, -1.7092, -1.9481, -1.1324, 0.2997, 0.2119, 0.2335};
const float _w2l [] = {0.8461, -0.6969, 0.2505, 1.9292, 1.4717, 1.2795, 0.5216, -0.6994, 1.3517, -0.8742};

const float _b2u = -0.4377;
const float _b2l = 0.7693;

const float _xoffset [InputNodes] = {220, 423, 218, 330};
const float _gain [InputNodes] = {0.0240963855421687, 0.0224719101123596, 0.0155038759689922, 0.0144927536231884};
const float _ymin = -1.0;

float _input[InputNodes];
float _hidden[HiddenNodes];
float _outputU = 0.0;
float _outputL = 0.0;
float _norm_input[InputNodes];
float _accum;

void setup() {
  Serial.begin(9600);
}

void loop() {
  nn_compute();
  delay(1000);
}

//Fungsi nn_compute (Forward Propagation)
void nn_compute() {
  //Pembacaan Sensor
  _input[0] = analogRead(A0);
  _input[1] = analogRead(A1);
  _input[2] = analogRead(A2);
  _input[3] = analogRead(A3);

  //Normalisasi Input
  int i, j;
  for (i = 0; i < InputNodes; i++) {
    _norm_input[i] = _input[i] - _xoffset[i];
    _norm_input[i] = _norm_input[i] * _gain[i];
    _norm_input[i] = _norm_input[i] + _ymin;
  }

  //Pengkosongan Memor (Hidden Layer)
  for (i = 0; i < HiddenNodes; i++) {
    _hidden[i] = 0;
  }

  //Perhitungan Hidden Layer
  for (i = 0; i < HiddenNodes; i++) {
    for (j = 0; j < InputNodes; j++) {
      _hidden[i] += _norm_input[j] * _w1[i][j];
    }
    _hidden[i] = (2.0 / (1.0 + exp (-(2 * (_hidden[i] + _b1[i]))))) - 1;
  }

  //Perhitungan Output Layer (q1)
  _accum = 0;
  for (i = 0; i < HiddenNodes; i++) {
    _accum += _hidden[i] * _w2u[i];
  }
  _outputU = 1.0 / (1.0 + exp(-(_accum + _b2u)));

    //Perhitungan Output Layer (q2)
  _accum = 0;
  for (i = 0; i < HiddenNodes; i++) {
    _accum += _hidden[i] * _w2l[i];
  }
  _outputL = 1.0 / (1.0 + exp(-(_accum + _b2l)));

//  Serial.print(_outputU);
//  Serial.print('\t');
//  Serial.println(_outputL);

  int kelas = 0;
  if (_outputU < 5.0 && _outputL >= 5.0) kelas = 1;       //APEK
  else if (_outputU >= 5.0 && _outputL < 5.0) kelas = 2;  //PESTISIDA
  else if (_outputU >= 5.0 && _outputL >= 5.0) kelas = 0;  //NORMAL

  int moisturePercentage = map(analogRead(A4), 540, 225, 0, 100);

  Serial.print('{');
  Serial.print(kelas);
  Serial.print(',');
  Serial.print(moisturePercentage);
  Serial.println('}');
}