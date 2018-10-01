#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <sstream>

#include <chrono>

int convo3x3(int A[3][3], int B[3][3]) {
	int tot = 0;
	for (int i = 0; i<3; i++)
		for (int j = 0; j<3; j++)
			tot += A[i][j] * B[i][j];

	return tot;
}

int convo5x5(int A[5][5], int B[5][5]) {
	int tot = 0;
	for (int i = 0; i<5; i++)
		for (int j = 0; j<5; j++)
			tot += A[i][j] * B[i][j];

	return tot;
}

int mean_rgb(cv::Vec3b pixel) {
	return (pixel.val[0] + pixel.val[1] + pixel.val[2]) / 3;
}

cv::Vec3b abs_rgb(cv::Vec3b vec) {
	return cv::Vec3b(std::abs(vec.val[0]), std::abs(vec.val[1]), std::abs(vec.val[2]));
}

int min_3(int a, int b, int c) {
	return std::min(a, std::min(b, c));
}

int argmin(int tab[], int length) {
	int index = 0;

	for (int i = 0; i < length; i++)
	{
		if (tab[i] < tab[index])
			index = i;
	}
	return index;
}

int argmin_horizontal(int **tab, int cols, int rows) {
	int index = 0;

	for (int i = 0; i < rows; i++)
	{
		if (tab[i][cols - 1] < tab[index][cols - 1])
			index = i;
	}
	return index;
}

int number_seams_to_remove(cv::Mat mask) {

	int max = 0;
	for (int y = 0; y<mask.rows; y++) {
		int count = 0;
		for (int x = 0; x<mask.cols; x++) {
			if (mask.at<uchar>(y, x) == 255)
				count++;
		}
		if (count > max)
			max = count;
	}
	return max;
}

cv::Mat gaussian_blur(cv::Mat image) {
	cv::Mat cop = image.clone();
	int conv[5][5] = { 1,4,7,4,1, 4,16,26,16,4, 7,26,41,26,7, 4,16,26,16,4, 1,4,7,4,1 };

	image.forEach<uchar>([image, &conv, &cop](uchar &p, const int * position) -> void {
		int y = position[0];
		int x = position[1];

		int voisinnage[5][5] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
		int i = 0;
		for (int yp = y - 1; yp<y + 2; yp++) {
			int j = 0;
			for (int xp = x - 1; xp < x + 2; xp++) {
				if (xp >= 0 && xp<image.cols && yp >= 0 && yp <image.rows) {
					voisinnage[j][i] = image.at<uchar>(yp, xp);
				}
				j++;
			}
			i++;
		}
		int c = convo5x5(conv, voisinnage) / 273;
		cop.at<uchar>(y, x) = c > 255 ? 255 : c;

	});
	return cop;
}

void sobel(cv::Mat img_gray, cv::Mat img_to) {
	int convx[3][3] = { -3,0,3, -10,0,10, -3,0,3 };
	int convy[3][3] = { -3,-10,-3, 0,0,0, 3,10,3 };

	img_gray.forEach<uchar>([img_gray, &convx, &convy, &img_to](uchar &p, const int * position) -> void {
		int y = position[0];
		int x = position[1];

		int voisinnage[3][3] = { 0,0,0,0,0,0,0,0,0 };
		int i = 0;
		for (int yp = y - 1; yp<y + 2; yp++) {
			int j = 0;
			for (int xp = x - 1; xp < x + 2; xp++) {
				if (xp >= 0 && xp<img_gray.cols && yp >= 0 && yp <img_gray.rows) {
					voisinnage[j][i] = img_gray.at<uchar>(yp, xp);
				}
				j++;
			}
			i++;
		}
		int Gx = convo3x3(convx, voisinnage);
		int Gy = convo3x3(convy, voisinnage);
		int G = sqrt(pow(Gx, 2) + pow(Gy, 2));
		img_to.at<uchar>(y, x) = G > 255 ? 255 : G;

	});
}

void sobel_mask(cv::Mat img_gray, cv::Mat img_to, cv::Mat img_mask,int coef) {
	int convx[3][3] = { -3,0,3, -10,0,10, -3,0,3 };
	int convy[3][3] = { -3,-10,-3, 0,0,0, 3,10,3 };

	img_gray.forEach<uchar>([img_gray, &convx, &convy, &img_to, &img_mask,&coef](uchar &p, const int * position) -> void {
		int y = position[0];
		int x = position[1];

		int voisinnage[3][3] = { 0,0,0,0,0,0,0,0,0 };
		int i = 0;
		for (int yp = y - 1; yp<y + 2; yp++) {
			int j = 0;
			for (int xp = x - 1; xp < x + 2; xp++) {
				if (xp >= 0 && xp<img_gray.cols && yp >= 0 && yp <img_gray.rows) {
					voisinnage[j][i] = img_gray.at<uchar>(yp, xp);
				}
				j++;
			}
			i++;
		}
		int Gx = convo3x3(convx, voisinnage);
		int Gy = convo3x3(convy, voisinnage);
		int G = sqrt(pow(Gx, 2) + pow(Gy, 2));
		//img_to.at<int>(y, x) = ((int)img_mask.at<uchar>(y, x) == 255) ? G*-1000 : G;
		img_to.at<int>(y, x) = ((int)img_mask.at<uchar>(y, x) >= 10) ? G*coef : G;
	});
}

template <typename T>
int*  seams_vertical(cv::Mat mat) {
	int** cop = new int*[mat.rows];
	for (int i = 0; i < mat.rows; ++i)
		cop[i] = new int[mat.cols];

	for (int y = 0; y < mat.rows; y++) {
		for (int x = 0; x < mat.cols; x++) {
			if (y == 0)
				cop[y][x] = mat.at<T>(y, x);
			else if (x == 0)
				cop[y][x] = mat.at<T>(y, x) + std::min(cop[y - 1][x], cop[y - 1][x + 1]);
			else if (x == mat.cols - 1)
				cop[y][x] = mat.at<T>(y, x) + std::min(cop[y - 1][x], cop[y - 1][x - 1]);
			else
				cop[y][x] = mat.at<T>(y, x) + min_3(cop[y - 1][x + 1], cop[y - 1][x], cop[y - 1][x - 1]);
		}
	}
	int argmin_end = argmin(cop[mat.rows - 1], mat.cols);

	//Only need to keep track of x's as y = index(x)
	//When we go through array, just keep track of index
	//Don't forget to free 
	int* seams_x = new int[mat.rows]();
	seams_x[mat.rows - 1] = argmin_end;

	int x = argmin_end;
	int dx[3] = { -1,0,1 };
	int values[3] = {};

	for (int y = mat.rows - 1; y > 0; y--) {
		int center = cop[y - 1][x];
		if (x == 0) {
			values[0] = 250000;
			values[1] = center;
			values[2] = cop[y - 1][x + 1];
		}
		else if (x == mat.cols - 1) {
			values[0] = cop[y - 1][x - 1];
			values[1] = center;
			values[2] = 250000;
		}
		else {
			values[0] = cop[y - 1][x - 1];
			values[1] = center;
			values[2] = cop[y - 1][x + 1];
		}
		x += dx[argmin(values, 3)];
		seams_x[y - 1] = x;
	}

	for (int i = 0; i < mat.rows; ++i)
		delete[] cop[i];
	delete[] cop;
	return seams_x;
}

template <typename T>
int*  seams_horizontal(cv::Mat mat) {
	int** cop = new int*[mat.rows];
	for (int i = 0; i < mat.rows; ++i)
		cop[i] = new int[mat.cols];

	for (int x = 0; x < mat.cols; x++) {
		for (int y = 0; y < mat.rows; y++) {
			if (x == 0)
				cop[y][x] = mat.at<T>(y, x);
			else if (y == 0)
				cop[y][x] = mat.at<T>(y, x) + std::min(cop[y][x - 1], cop[y + 1][x - 1]);
			else if (y == mat.rows - 1)
				cop[y][x] = mat.at<T>(y, x) + std::min(cop[y][x - 1], cop[y - 1][x - 1]);
			else
				cop[y][x] = mat.at<T>(y, x) + min_3(cop[y - 1][x - 1], cop[y][x - 1], cop[y + 1][x - 1]);
		}
	}

	int argmin_end = argmin_horizontal(cop, mat.cols, mat.rows);

	//Only need to keep track of y's as x = index(y)
	//When we go through array, just keep track of index
	//Don't forget to free 
	int* seams_y = new int[mat.cols]();
	seams_y[mat.cols - 1] = argmin_end;

	int y = argmin_end;
	int dx[3] = { -1,0,1 };
	int values[3] = {};

	for (int x = mat.cols - 1; x > 0; x--) {
		int center = cop[y][x - 1];
		if (y == 0) {
			//std::cout << "0"<< "(" << x << "," << y << ")" << std::endl;
			values[0] = 250000;
			values[1] = center;
			values[2] = cop[y + 1][x - 1];
		}
		else if (y == mat.rows - 1) {
			// std::cout << "mat.rows-1"<< "(" << x << "," << y << ")"  << std::endl;
			values[0] = cop[y - 1][x - 1];
			values[1] = center;
			values[2] = 250000;
		}
		else {
			// std::cout << "normal"<< "(" << x << "," << y << ")"  << std::endl;
			values[0] = cop[y - 1][x - 1];
			values[1] = center;
			values[2] = cop[y + 1][x - 1];
		}
		y += dx[argmin(values, 3)];
		seams_y[x - 1] = y;
	}

	for (int i = 0; i < mat.rows; ++i)
		delete[] cop[i];
	delete[] cop;
	return seams_y;
}

cv::Mat highlight_seams_horizontal(cv::Mat img, int list[]) {

	cv::Mat cop = img.clone();

	for (int x = 0; x < img.cols; x++) {
		cop.at<cv::Vec3b>(list[x], x)[0] = 0;
		cop.at<cv::Vec3b>(list[x], x)[1] = 0;
		cop.at<cv::Vec3b>(list[x], x)[2] = 255;

	}
	return cop;
}

cv::Mat highlight_seams_vertical(cv::Mat img, int list[]) {

	cv::Mat cop = img.clone();

	for (int y = 0; y < img.rows; y++) {
		cop.at<cv::Vec3b>(y, list[y])[0] = 0;
		cop.at<cv::Vec3b>(y, list[y])[1] = 0;
		cop.at<cv::Vec3b>(y, list[y])[2] = 255;

	}
	return cop;
}

template <typename T>
cv::Mat move_im_horizontal(cv::Mat mat, int ys[]) {
	//for RGB : color = CV_8UC3
	//for GrayScale color = CV_8UC1
	cv::Mat cop(mat.rows - 1, mat.cols, mat.type());

	for (int x = 0; x < cop.cols; x++) {
		for (int y = 0; y < cop.rows; y++) {
			if (y < ys[x])
				cop.at<T>(y, x) = mat.at<T>(y, x);
			else
				cop.at<T>(y, x) = mat.at<T>(y + 1, x);
		}
	}
	return cop;
}

template <typename T>
cv::Mat move_im_vertical(cv::Mat mat, int xs[]) {
	//for RGB : color = CV_8UC3
	//for GrayScale color = CV_8UC1
	cv::Mat cop(mat.rows, mat.cols - 1, mat.type());

	for (int y = 0; y < cop.rows; y++) {
		for (int x = 0; x < cop.cols; x++) {
			if (x < xs[y])
				cop.at<T>(y, x) = mat.at<T>(y, x);
			else
				cop.at<T>(y, x) = mat.at<T>(y, x + 1);
		}
	}
	return cop;
}

template <typename T>
cv::Mat insert_seams_horizontal(cv::Mat mat, int ys[]) {

	cv::Mat cop(mat.rows + 1, mat.cols, mat.type());

	for (int x = 0; x < cop.cols; x++) {
		for (int y = 0; y < cop.rows; y++) {
			if (y < ys[x])
				cop.at<T>(y, x) = mat.at<T>(y, x);
			else if (y == ys[x]) {
				cop.at<T>(y, x) = mat.at<T>(y, x);
				cop.at<T>(y + 1, x) = y == mat.rows - 1 ? mat.at<T>(y, x) : mat.at<T>(y, x)*0.5 + mat.at<T>(y + 1, x)*0.5;

			}
			else if (y > ys[x] + 1)
				cop.at<T>(y, x) = mat.at<T>(y - 1, x);
		}
	}
	return cop;
}

template <typename T>
cv::Mat insert_seams_vertical(cv::Mat mat, int xs[]) {

	cv::Mat cop(mat.rows, mat.cols + 1, mat.type());

	for (int y = 0; y < cop.rows; y++) {
		for (int x = 0; x < cop.cols; x++) {
			if (x < xs[y])
				cop.at<T>(y, x) = mat.at<T>(y, x);
			else if (x == xs[y]) {
				cop.at<T>(y, x) = mat.at<T>(y, x);
				cop.at<T>(y, x + 1) = mat.at<T>(y, x)*0.5 + mat.at<T>(y, x + 1)*0.5;
			}
			else if (x > xs[y] + 1)
				cop.at<T>(y, x) = mat.at<T>(y, x - 1);
		}
	}
	return cop;
}

template <typename T>
cv::Mat insert_seams_gradient_horizontal(cv::Mat mat, int ys[]) {

	cv::Mat cop(mat.rows + 1, mat.cols, mat.type());

	for (int x = 0; x < cop.cols; x++) {
		for (int y = 0; y < cop.rows; y++) {
			if (y < ys[x])
				cop.at<T>(y, x) = mat.at<T>(y, x);
			else if (y == ys[x]) {
				cop.at<T>(y, x) = 255;
				cop.at<T>(y + 1, x) = mat.at<T>(y, x)*0.5 + mat.at<T>(y + 1, x)*0.5;
			}
			else if (y > ys[x] + 1)
				cop.at<T>(y, x) = mat.at<T>(y - 1, x);
		}
	}
	return cop;
}

template <typename T>
cv::Mat insert_seams_gradient_vertical(cv::Mat mat, int xs[]) {

	cv::Mat cop(mat.rows, mat.cols + 1, mat.type());

	for (int y = 0; y < cop.rows; y++) {
		for (int x = 0; x < cop.cols; x++) {
			if (x < xs[y])
				cop.at<T>(y, x) = mat.at<T>(y, x);
			else if (x == xs[y]) {
				cop.at<T>(y, x) = 255;
				cop.at<T>(y, x + 1) = mat.at<T>(y, x)*0.5 + mat.at<T>(y, x + 1)*0.5;
			}
			else if (x > xs[y] + 1)
				cop.at<T>(y, x) = mat.at<T>(y, x - 1);
		}
	}
	return cop;
}

template <typename T>
cv::Mat add_vertical(int n, cv::Mat image, cv::Mat gradient) {

	cv::Mat result = image.clone();
	for (int i = 0; i < n; i++) {

		int* list_x_seams = seams_vertical<T>(gradient);
		result = insert_seams_vertical<cv::Vec3b>(result, list_x_seams);
		gradient = insert_seams_gradient_vertical<T>(gradient, list_x_seams);

		/*std::stringstream gradient_name, result_name;
		gradient_name << "./gradient/gradientADD" << i << ".png";
		result_name << "./result/resultADD" << i << ".png";
		cv::imwrite(gradient_name.str().c_str(), gradient);
		cv::imwrite(result_name.str().c_str(), result);*/


		free(list_x_seams);

		std::cout << "Adding vertical seams : " << int(float(i) / n * 100) << "%" << "\r" << std::flush;
	}
	std::cout << std::endl;
	return result;
}

void sobel_fracture_vertical(cv::Mat gradient, cv::Mat image, int* list_seams) {
	int xi;
	int convx[3][3] = { -3,0,3, -10,0,10, -3,0,3 };
	int convy[3][3] = { -3,-10,-3, 0,0,0, 3,10,3 };
	for (int y = 0; y < gradient.rows; y++) {
		xi = list_seams[y];
		for (int x = xi - 2; x <= xi + 2; x++) {
			//We check that the pixel is in the image
			if (x<0 || x >= gradient.cols)
				continue;
			//std::cout <<"(x,y)" << x << "," << y << std::endl;
			int voisinnage[3][3] = { 0,0,0,0,0,0,0,0,0 };
			int i = 0;
			for (int yp = y - 1; yp<y + 2; yp++) {
				int j = 0;
				for (int xp = x - 1; xp < x + 2; xp++) {
					if (xp >= 0 && xp<gradient.cols && yp >= 0 && yp <gradient.rows) {
						voisinnage[j][i] = image.at<uchar>(yp, xp);
					}
					j++;
					//std::cout << "(xp,yp)" << xp << "," << yp << std::endl;

				}
				i++;
			}
			int Gx = convo3x3(convx, voisinnage);
			int Gy = convo3x3(convy, voisinnage);
			int G = sqrt(pow(Gx, 2) + pow(Gy, 2));
			gradient.at<uchar>(y, x) = G > 255 ? 255 : G;
			//std::cout << "set" << std::endl;
		}
	}

}

void sobel_fracture_vertical_mask(cv::Mat gradient, cv::Mat image, int* list_seams,cv::Mat mask, int coef) {
	int xi;
	int convx[3][3] = { -3,0,3, -10,0,10, -3,0,3 };
	int convy[3][3] = { -3,-10,-3, 0,0,0, 3,10,3 };
	for (int y = 0; y < gradient.rows; y++) {
		xi = list_seams[y];
		for (int x = xi - 2; x <= xi + 2; x++) {
			//We check that the pixel is in the image
			if (x<0 || x >= gradient.cols)
				continue;
			//std::cout <<"(x,y)" << x << "," << y << std::endl;
			int voisinnage[3][3] = { 0,0,0,0,0,0,0,0,0 };
			int i = 0;
			for (int yp = y - 1; yp<y + 2; yp++) {
				int j = 0;
				for (int xp = x - 1; xp < x + 2; xp++) {
					if (xp >= 0 && xp<gradient.cols && yp >= 0 && yp <gradient.rows) {
						voisinnage[j][i] = image.at<uchar>(yp, xp);
					}
					j++;
					//std::cout << "(xp,yp)" << xp << "," << yp << std::endl;

				}
				i++;
			}
			int Gx = convo3x3(convx, voisinnage);
			int Gy = convo3x3(convy, voisinnage);
			int G = sqrt(pow(Gx, 2) + pow(Gy, 2));
			gradient.at<int>(y, x) = ((int)mask.at<uchar>(y, x) == 255) ? G*coef : G;
			//std::cout << "set" << std::endl;
		}
	}

}

cv::Mat add_horizontal(int n, cv::Mat image, cv::Mat gradient) {

	cv::Mat result = image.clone();
	for (int i = 0; i < n; i++) {

		// std::cout<<"start" <<std::endl;
		int* list_x_seams = seams_horizontal<uchar>(gradient);
		// std::cout<< "seams found" << std::endl;
		result = insert_seams_horizontal<cv::Vec3b>(result, list_x_seams);
		// std::cout<< "inserted result"<<std::endl;
		gradient = insert_seams_gradient_horizontal<uchar>(gradient, list_x_seams);
		// std::cout<< "inserted grad"<<std::endl;

		free(list_x_seams);

		std::cout << "Adding horizontal seams : " << int(float(i) / n * 100) << "%" << "\r" << std::flush;
	}
	std::cout << std::endl;
	return result;
}

cv::Mat remove_horizontal(int n, cv::Mat image, cv::Mat gradient) {

	cv::Mat result = image.clone();
	for (int i = 0; i < n; i++) {

		int* list_x_seams = seams_horizontal<uchar>(gradient);
		result = move_im_horizontal<cv::Vec3b>(result, list_x_seams);
		gradient = move_im_horizontal<uchar>(gradient, list_x_seams);
		free(list_x_seams);

		std::cout << "Removing horizontal seams : " << int(float(i) / n * 100) << "%" << "\r" << std::flush;
	}
	std::cout << std::endl;
	return result;
}

template <typename T>
cv::Mat remove_vertical(int n, cv::Mat image, cv::Mat gradient, cv::Mat image_gray) {
	cv::Mat result = image.clone();
	for (int i = 0; i < n; i++) {

		int* list_x_seams = seams_vertical<T>(gradient);
		result = move_im_vertical<cv::Vec3b>(result, list_x_seams);
		gradient = move_im_vertical<T>(gradient, list_x_seams);
		image_gray = move_im_vertical<uchar>(image_gray, list_x_seams);
		sobel_fracture_vertical(gradient,image_gray, list_x_seams);

				free(list_x_seams);
		std::cout << "Removing vertical seams : " << int(float(i) / n * 100) << "%" << "\r" << std::flush;
	}
	std::cout << std::endl;
	return result;
}

template <typename T>
cv::Mat remove_vertical_mask(cv::Mat image, cv::Mat gradient, cv::Mat image_gray, cv::Mat mask, int* number) {
	cv::Mat result = image.clone();

	while (nbr_points_mask(mask) > 0 ){
		std::cout << nbr_points_mask(mask) << std::endl;
		int* list_x_seams = seams_vertical<T>(gradient);
		result = move_im_vertical<cv::Vec3b>(result, list_x_seams);
		gradient = move_im_vertical<T>(gradient, list_x_seams);
		image_gray = move_im_vertical<uchar>(image_gray, list_x_seams);
		mask = move_im_vertical<uchar>(mask, list_x_seams);

		/*std::stringstream gradient_name, result_name, gray_name;
		gradient_name << "gradient\\gradientRM" << *number << ".png";
		result_name << "result\\resultRM" << *number << ".png";
		gray_name << "mask\\mask" << *number << ".png";
		cv::imwrite(gradient_name.str().c_str(), gradient);
		cv::imwrite(gray_name.str().c_str(), mask);
		cv::imwrite(result_name.str().c_str(), highlight_seams_vertical(result, list_x_seams));*/

		free(list_x_seams);
		*number = *number + 1;
		//std::cout << "Removing vertical seams : "<< "\r" << std::flush;
	}
	std::cout << std::endl;
	return result;
}

template <typename T>
cv::Mat remove_vertical_accent(int n, cv::Mat image, cv::Mat gradient, cv::Mat image_gray, cv::Mat mask, int coef) {
	cv::Mat result = image.clone();
	for (int i = 0; i < n; i++) {

		int* list_x_seams = seams_vertical<T>(gradient);
		result = move_im_vertical<cv::Vec3b>(result, list_x_seams);
		gradient = move_im_vertical<T>(gradient, list_x_seams);
		image_gray = move_im_vertical<uchar>(image_gray, list_x_seams);
		mask = move_im_vertical<uchar>(mask, list_x_seams);
		sobel_fracture_vertical_mask(gradient, image_gray, list_x_seams,mask,coef);
		
		/*std::stringstream gradient_name, result_name, gray_name;
		gradient_name << "gradient\\gradient" << i << ".png";
		result_name << "result\\result" << i << ".png";
		gray_name << "gray\\gray" << i << ".png";
		cv::imwrite(gradient_name.str().c_str(), gradient);
		cv::imwrite(gray_name.str().c_str(), image_gray);
		cv::imwrite(result_name.str().c_str(), highlight_seams_vertical(result,list_x_seams));*/
	
		free(list_x_seams);
		std::cout << "Removing vertical seams : " << int(float(i) / n * 100) << "%" << "\r" << std::flush;
	}
	std::cout << std::endl;
	return result;
}

int nbr_points_mask(cv::Mat mask) {
	int i= 0;

	mask.forEach<uchar>([mask,&i](uchar &p, const int * position) -> void {
		if (p > 100)
			i++;
	});
	return i;
}

int main(int argc, char** argv) {

	cv::Mat image, image_gray, gradient, gradient_before, result, result_x;
	image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if (!image.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}


	cv::cvtColor(image, image_gray, CV_RGB2GRAY);

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	//gradient_before = image_gray.clone();
	//	gradient = gaussian_blur(gradient_before);

	gradient = image_gray.clone();

	sobel(image_gray, gradient);

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> time_span = t2 - t1;
	std::cout << "Sobel took :" << time_span.count() << " milliseconds." << std::endl;
	result = image.clone();


	if (std::strcmp(argv[2], "resize") == 0) {

		//Timing sobel execution time		

		t1 = std::chrono::high_resolution_clock::now();

		int image_x = image.cols;
		int image_y = image.rows;

		int x_to_remove = image_x - atoi(argv[3]);
		int y_to_remove = image_y - atoi(argv[4]);
		std::cout << "x_to_remove" << x_to_remove << "y_to_remove" << y_to_remove << std::endl;

		if (x_to_remove < 0)
			result_x = add_vertical<uchar>(-x_to_remove, image, gradient);
		else
			result_x = remove_vertical<uchar>(x_to_remove, image, gradient, image_gray);

		cv::cvtColor(result_x, image_gray, CV_RGB2GRAY);
		//gradient_before = image_gray.clone();
		//gradient = gaussian_blur(gradient_before);

		gradient = image_gray.clone();
		//cv::GaussianBlur(image_gray, gradient, cv::Size(11, 11), 0);
		sobel(image_gray, gradient);

		if (y_to_remove < 0)
			result = add_horizontal(-y_to_remove, result_x, gradient);
		else
			result = remove_horizontal(y_to_remove, result_x, gradient);

		std::cout << std::endl;
		t2 = std::chrono::high_resolution_clock::now();
		time_span = t2 - t1;
		std::cout << "All operations after took : " << time_span.count() << " milliseconds." << std::endl;
	}
	else if (std::strcmp(argv[2], "mask") == 0) {
		cv::Mat mask = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);

		if (!mask.data) {
			std::cout << "Could not open or find the mask image" << std::endl;
			return -1;
		}
		cv::Mat grad(result.rows, result.cols, CV_32SC1, cv::Scalar(0));

		sobel_mask(image_gray, grad, mask,-1000);
		std::cout<<"mask sobel done" <<std::endl;
		int number = 0;
		int j = number_seams_to_remove(mask);

		//result = remove_vertical_mask<int>(number, result, grad, image_gray,mask,-1000);
		result = remove_vertical_mask<int>(result, grad, image_gray,mask,&number);

		std::cout << number << " " << j << std::endl;

		cv::cvtColor(result, image_gray, CV_RGB2GRAY);
		grad = image_gray.clone();
		sobel(image_gray, grad);

		result = add_vertical<uchar>(number, result, grad);

	}
	else if (std::strcmp(argv[2], "accent") == 0) {
		cv::Mat mask = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);

		if (!mask.data) {
			std::cout << "Could not open or find the mask image" << std::endl;
			return -1;
		}

		//Timing sobel execution time		

		t1 = std::chrono::high_resolution_clock::now();

		int image_x = image.cols;
		int image_y = image.rows;

		int x_to_remove = image_x - atoi(argv[4]);
		int y_to_remove = image_y - atoi(argv[5]);
		std::cout << "x_to_remove" << x_to_remove << "y_to_remove" << y_to_remove << std::endl;

		cv::Mat grad(result.rows, result.cols, CV_32SC1, cv::Scalar(0));
		sobel_mask(image_gray, grad, mask, 100);


		if (x_to_remove < 0)
			result_x = add_vertical<int>(-x_to_remove, image, grad);
		else
			result_x = remove_vertical_accent<int>(x_to_remove, image, grad, image_gray,mask,100);

		cv::cvtColor(result_x, image_gray, CV_RGB2GRAY);
		cv::Mat grad2(image_gray.rows, image_gray.cols, CV_32SC1, cv::Scalar(0));
		sobel_mask(image_gray, grad2, mask, 100);

		if (y_to_remove < 0)
			result = add_horizontal(-y_to_remove, result_x, grad2);
		else
			result = remove_horizontal(y_to_remove, result_x, grad2);

		std::cout << std::endl;
		t2 = std::chrono::high_resolution_clock::now();
		time_span = t2 - t1;
		std::cout << "All operations after took : " << time_span.count() << " milliseconds." << std::endl;
	}
	else {
		std::cout << "Choose correct mode: resize, mask" << std::endl;
		return -1;
	}

	cv::imwrite("resultMask.png", result);
	//cv::imwrite("result.png", result);

	cv::waitKey();


	return 0;
}

/*image storing
std::stringstream gradient_name,result_name;
gradient_name << "./gradient/gradient"<< i << ".png";
result_name << "./result/result"<< i << ".png" ;
cv::imwrite( gradient_name.str().c_str(), gradient );
cv::imwrite( result_name.str().c_str(), result );
*/