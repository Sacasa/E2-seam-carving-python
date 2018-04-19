#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>

#include <chrono>
 
int convo3x3(int A[3][3], int B[3][3]){
    int tot = 0;
    for(int i = 0; i<3 ;i++)
        for(int j = 0; j<3; j++)
            tot += A[i][j]*B[i][j];

    return tot;
}

int min_3(int a,int b,int c){
    return std::min(a,std::min(b,c));
}

int argmin(int tab[], int length){
    int index = 0;

    for(int i = 1; i < length; i++)
    {
        if(tab[i] < tab[index])
            index = i;              
    }
    return index;
}

void sobel(cv::Mat img_gray, cv::Mat img_to){
    int convx[3][3] = {-1,0,1, -2,0,2, -1,0,1};
    int convy[3][3] = {-1,-2,-1, 0,0,0, 1,2,1};

    /* Look at opencv forEach for multithreaded for earch loop*/
    for(int y=0; y<img_gray.rows; y++){
        for(int x=0; x<img_gray.cols; x++){
                

            int voisinnage[3][3] = {0,0,0,0,0,0,0,0,0};
            int i=0;
            for(int yp= y-1; yp<y+2; yp++){
                int j = 0;
                for(int xp = x-1; xp < x+2; xp++){
                    if(xp >= 0 && xp<img_gray.cols && yp >=0 && yp <img_gray.rows){
                        voisinnage[j][i] = img_gray.at<uchar>(yp,xp);
                    }
                    j++;
                }
                i++;
            }
            int Gx = convo3x3(convx,voisinnage);
            int Gy = convo3x3(convy,voisinnage);
            int G = sqrt(pow(Gx,2) + pow(Gy,2));
            img_to.at<uchar>(y,x) = G > 255 ? 255 : G;
        }
    }
    std::cout<< "Sobel done" <<std::endl;

}


int*  seams(cv::Mat mat){
    int cop[mat.rows][mat.cols] = { };

    for(int y = 0; y < mat.rows; y++){
        for(int x = 0; x < mat.cols; x++){
            if(y==0)
                cop[y][x] = mat.at<uchar>(y,x);
            else if(x==0)
                cop[y][x] = mat.at<uchar>(y,x) + std::min(cop[y-1][x],cop[y-1][x+1]);
            else if(x == mat.cols-1)
                cop[y][x] = mat.at<uchar>(y,x) + std::min(cop[y-1][x],cop[y-1][x-1]);
            else
                cop[y][x] = mat.at<uchar>(y,x) + min_3(cop[y-1][x+1],cop[y-1][x],cop[y-1][x-1]) ; 
        }
    }

    int argmin_end = argmin(cop[mat.rows-1],mat.cols);
    
    //Only need to keep track of x's as y = index(x)
    //When we go through array, just keep track of index
    int* seams_x = new int[mat.rows]();
    seams_x[mat.rows-1] = argmin_end;

    int x = argmin_end;
    int dx[3] = {-1,0,1};
    int values[3] = {};

    for(int y = mat.rows-1; y > 0; y--){
        int center = cop[y-1][x];
        if(x==0){
            values[0] =250000 ;
            values[1] =center ;
            values[2] =cop[y-1][x+1];   
        }
        else if(x == mat.cols-1){
            values[0] = cop[y-1][x-1];
            values[1] =center;
            values[2] =250000;
        }
        else{
            values[0] =cop[y-1][x-1] ;
            values[1] =center ;
            values[2] =cop[y-1][x+1];
        }
        x += dx[argmin(values,3)];
        seams_x[y-1] = x;
    }
    return seams_x;
}

int main( int argc, char** argv ) {
  
  cv::Mat image, image_gray, image_to;
  image = cv::imread(argv[1] , CV_LOAD_IMAGE_COLOR);
  


  if(! image.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
  }

  cv::cvtColor(image, image_gray, CV_RGB2GRAY);
  
  image_to = image_gray.clone();


  //Timing sobel execution time

  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  sobel(image_gray,image_to);

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> time_span = t2 - t1;
  std::cout << "It took me " << time_span.count() << " milliseconds." << std::endl;


  //Showing Sobel and then original image
  
  cv::imshow( "Sobel", image_to );
  cv::waitKey(0);

  cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Display window", image );
  
  cv::waitKey(0);
  return 0;
}