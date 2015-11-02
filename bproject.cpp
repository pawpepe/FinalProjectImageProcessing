/**
    PROJETO FINAL B
    SCC0251 ­ PDI PROFESSOR : MOACIR P.

    Leonardo Fachetti   #USP 6878870 
    Anna Paula P. Maule #USP 4624650
**/


#include <string>
#include <stdio.h>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

/*
	Comparação para a função de ordenação sort
 */
bool mySort(pair<Point2f, double> a, pair<Point2f, double> b) {
  return a.second > b.second;
}

/*
	Encontra o ponto central dado um conunto de pontos.
	Na segunda iteração exclui os pontos de maior distancia desse ponto central previamente calculado e o re-calcula
	Entrada
		Vetor de pontos
		Numero de iterações
	Saida
		Ponto central
 */

Point2f average(vector<Point2f> v, int it) {
	Point2f av;
  	vector< pair<Point2f, double> > points;

	while(it--) {
		if(!points.empty()) {
			v.clear();
			for (int i = points.size()*0.4; i < (signed) points.size(); i++)  {
				v.push_back(Point2f(points[i].first.x, points[i].first.y));
			}
		} else {
			points.clear();
		}

		int sumX = 0, sumY = 0;

		// Calcula o valor médio de x e y
		for(int i = 0 ; i < (signed) v.size(); i++) {
			sumX +=v[i].x;
			sumY +=v[i].y;
		}
		av.x = sumX/v.size();
		av.y = sumY/v.size();

		if(!it) {
			break;
		}

		for (int i = 0; i < (signed) v.size(); i++)  {
			points.push_back(make_pair(v[i], sqrt(pow(abs(v[i].x - av.x), 2) + pow(abs(v[i].y - av.y), 2))));
		}

		sort(points.begin(), points.end(), mySort);
	}
	return av;
}

void readme() {
	cout << "Entrada invalida. \n\tUso: ./PROG_NAME <OBJ_BUSCADO_1> <OBJ_BUSCADO_2> <PASTA_COM_AS_CENAS> <NUM_IMAGENS_NA_PASTA>" << endl;
}

int main(int argc, char** argv){
	Mat output, rec, descriptor, descriptor1, descriptor1_2, descriptor2;
	FlannBasedMatcher matcher;
	vector<vector<DMatch > > matches;
	vector<DMatch > good_matches;
	vector<Point2f> obj, scene;

	if(argc != 5) {
		readme();
		return -1;
	}

	Mat img1 = imread(argv[1], 1), img1_2 = imread(argv[2], 1), img2;

	// Borra as imagem
	GaussianBlur(img1, img1, Size(3,3), 0, 0, BORDER_DEFAULT);
	GaussianBlur(img1_2, img1_2, Size(3,3), 0, 0, BORDER_DEFAULT);

	vector<Mat> channels;
	// Separa os canais HSV para a imagem 1
	split(img1, channels);
	// Usa o canal H
	img1 = channels[0].clone();

	split(img1_2, channels);
	// Usa o canal H
	img1_2 = channels[0].clone();

	int minHessian = 400;
	vector<KeyPoint> keypoints1, keypoints1_2, keypoints2;
	SurfDescriptorExtractor extractor;
	SurfFeatureDetector detector(minHessian);

	//Detecta os keypoints da  primeira imagem usando SURF
	detector.detect(img1, keypoints1);

	//Calcula o descritor pra primeira imagem
	extractor.compute(img1, keypoints1, descriptor1);

	//Detecta os keypoints da  primeira imagem usando SURF
	detector.detect(img1_2, keypoints1_2);

	//Calcula o descritor pra primeira imagem
	extractor.compute(img1, keypoints1_2, descriptor1_2);

	for (int img = 1; img <= atoi(argv[4]); img++)  {
		string sceneName = argv[3];

		sceneName += "/test";
		if(img < 10)
			sceneName += "0";
		sceneName += to_string(img) + ".jpg";

		img2 = imread(sceneName, 1);

		if (img1.empty() || img1_2.empty() || img2.empty()) {
			printf("Can't read one of the images 1\n");
			return -1;
		}

		rec = img2.clone();

		// Borra a imagem
		GaussianBlur(img2, img2, Size(3,3), 0, 0, BORDER_DEFAULT);

		// Separa os canais HSV para a imagem 2
		split(img2, channels);
		// Usa o canal H
		img2 = channels[0].clone();

		//Detecta os keypoints da segunda imagem usando SURF
		detector.detect(img2, keypoints2);

		//Calcula o descritor pra segunda imagem
		extractor.compute(img2, keypoints2, descriptor2);

		matcher.knnMatch(descriptor1, descriptor2, matches, 2);

		for (int i = 0; i < min(descriptor2.rows - 1, (int)matches.size()); i++) {
			if ((matches[i][0].distance < 0.6*(matches[i][4].distance)) && ((int)matches[i].size() <= 2 && (int)matches[i].size()>0)) {
				good_matches.push_back(matches[i][0]);
			}
		}

		matcher.knnMatch(descriptor1_2, descriptor2, matches, 2);

		for (int i = 0; i < min(descriptor2.rows - 1, (int)matches.size()); i++) {
			if ((matches[i][0].distance < 0.6*(matches[i][4].distance)) && ((int)matches[i].size() <= 2 && (int)matches[i].size()>0)) {
				good_matches.push_back(matches[i][0]);
			}
		}

		//Draw only "good" matches
		drawMatches(img1, keypoints1, img2, keypoints2, good_matches, output, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		if (good_matches.size() >= 4) {
			for (int i = 0; i < (signed) good_matches.size(); i++) {
				//Get the keypoints from the good matches
				obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
				scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
			}
		}

		Point2f av = average(scene, 3);					// Calcula o ponto medio

		// Desenha um retangulo em torno do ponto medio
	  	rectangle(rec, Point2f(av.x - 150, av.y - 150), Point2f(av.x + 150, av.y + 150), Scalar(0,255,0), 3);

		// namedWindow("KeyPoints", CV_WINDOW_NORMAL);		// Cria uma janela para mostrar a imagem
		// imshow("KeyPoints", output);                   	// Mostra as duas imagens e seus keypoints
		// waitKey();

		namedWindow("Find Image", CV_WINDOW_NORMAL);	// Cria uma janela para mostrar a imagem
		imshow("Find Image", rec);                   	// Mostra a imagem com o retangulo sobre o objeto
		waitKey();

		good_matches.clear();
		obj.clear();
		scene.clear();
	}
	return 0;
}
