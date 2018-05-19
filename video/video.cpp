#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <sstream>

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

int  number_seams_to_remove(cv::Mat mask){

    int max = 0;
    for(int y=0; y<mask.rows; y++){
        int count = 0;
        for(int x=0; x<mask.cols; x++){
            if(mask.at<uchar>(y,x) == 255)
                count++;
        }
        if(count > max)
            max = count;
    }
    return max;
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
}

void sobel_mask(cv::Mat img_gray, cv::Mat img_to, cv::Mat img_mask){
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
            
            img_to.at<int>(y,x) = ((int)img_mask.at<uchar>(y,x) == 255) ? G*-1000 : G;

        }
    }
}

template <typename T>
int*  seams(cv::Mat mat){
    int** cop = new int*[mat.rows];
    for (int i = 0; i < mat.rows; ++i)
        cop[i] = new int[mat.cols];

    for(int y = 0; y < mat.rows; y++){
        for(int x = 0; x < mat.cols; x++){
            if(y==0)
                cop[y][x] = mat.at<T>(y,x);
            else if(x==0)
                cop[y][x] = mat.at<T>(y,x) + std::min(cop[y-1][x],cop[y-1][x+1]);
            else if(x == mat.cols-1)
                cop[y][x] = mat.at<T>(y,x) + std::min(cop[y-1][x],cop[y-1][x-1]);
            else
                cop[y][x] = mat.at<T>(y,x) + min_3(cop[y-1][x+1],cop[y-1][x],cop[y-1][x-1]) ; 
        }
    }

    int argmin_end = argmin(cop[mat.rows-1],mat.cols);
    
    //Only need to keep track of x's as y = index(x)
    //When we go through array, just keep track of index
    //Don't forget to free 
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

    for (int i = 0; i < mat.rows; ++i)
        delete [] cop[i];
    delete [] cop;
    return seams_x;
}

cv::Mat highlight_seams(cv::Mat img, int list[]){

    cv::Mat cop = img.clone();

    for(int y = 0; y < img.rows; y++){
        cop.at<cv::Vec3b>(y,list[y])[0] = 255;
        cop.at<cv::Vec3b>(y,list[y])[1] = 0;
        cop.at<cv::Vec3b>(y,list[y])[2] = 0;

    }
    return cop;
}

template <typename T>
cv::Mat move_im(cv::Mat mat, int xs[]){
    //for RGB : color = CV_8UC3
    //for GrayScale color = CV_8UC1
    cv::Mat cop(mat.rows,mat.cols-1,mat.type());

    for(int y = 0; y < cop.rows; y++){
        for(int x = 0; x < cop.cols; x++){
            if(x < xs[y])
                cop.at<T>(y,x) = mat.at<T>(y,x);
            else
                cop.at<T>(y,x) = mat.at<T>(y,x+1);
        }
    }
    return cop;
}

template <typename T>
cv::Mat insert_seams(cv::Mat mat, int xs[]){

    cv::Mat cop(mat.rows,mat.cols+1,mat.type());

    for(int y = 0; y < cop.rows; y++){
        for(int x = 0; x < cop.cols; x++){
            if(x < xs[y])
                cop.at<T>(y,x) = mat.at<T>(y,x);
            else if(x == xs[y]){
                cop.at<T>(y,x) = mat.at<T>(y,x);
                cop.at<T>(y,x+1) = mat.at<T>(y,x)*0.5 + mat.at<T>(y,x+1)*0.5;
            }
            else if(x > xs[y] +1)
                cop.at<T>(y,x) = mat.at<T>(y,x-1);
        }
    }
    return cop;
}

template <typename T>
cv::Mat insert_seams_gradient(cv::Mat mat, int xs[]){

    cv::Mat cop(mat.rows,mat.cols+1,mat.type());

    for(int y = 0; y < cop.rows; y++){
        for(int x = 0; x < cop.cols; x++){
            if(x < xs[y])
                cop.at<T>(y,x) = mat.at<T>(y,x);
            else if(x == xs[y]){
                cop.at<T>(y,x) = 255;
                cop.at<T>(y,x+1) = mat.at<T>(y,x)*0.5 + mat.at<T>(y,x+1)*0.5;
            }
            else if(x > xs[y] +1)
                cop.at<T>(y,x) = mat.at<T>(y,x-1);
        }
    }
    return cop;
}


int main( int argc, char** argv ) {
  
 cv::VideoCapture video_in(argv[1]); 
    
  // Check if camera opened successfully
  if(!video_in.isOpened()){
    std::cout << "Error opening video stream or file" << std::endl;
    return -1;
  }

  // Default resolution of the frame is obtained.The default resolution is system dependent. 
  int frame_width = video_in.get(CV_CAP_PROP_FRAME_WIDTH)-atoi(argv[2]); 
  int frame_height = video_in.get(CV_CAP_PROP_FRAME_HEIGHT); 
  int frame_count = video_in.get(CV_CAP_PROP_FRAME_COUNT);
  std::cout << "Video loaded : "<< frame_width + atoi(argv[2])<< "x" << frame_height << " , "<< frame_count << " frames at "<< video_in.get(CV_CAP_PROP_FPS)<< " fps"<<std::endl;

  // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file. 
  cv::VideoWriter video_out("outcpp.avi",CV_FOURCC('M','J','P','G'),video_in.get(CV_CAP_PROP_FPS), cv::Size(frame_width,frame_height)); 

  int j = 0;
  while(1){
    j++;
    cv::Mat frame, frame_gray, frame_gradient,frame_result;
    // capture frame-by-frame
    video_in >> frame;

    // If the frame is empty, break immediately
    if (frame.empty())
      break;

    cv::cvtColor(frame, frame_gray, CV_RGB2GRAY);
    frame_gradient = frame_gray.clone();

    sobel(frame_gray,frame_gradient);

    frame_result = frame.clone();

    for(int i = 0; i < atoi(argv[2]);i++){
      int* list_x_seams = seams<uchar>(frame_gradient);
      frame_result = move_im<cv::Vec3b>(frame_result, list_x_seams);
      frame_gradient = move_im<uchar>(frame_gradient, list_x_seams);
      free(list_x_seams);
    std::cout << "Video processing :" <<int(float(j)/frame_count * 100)<<"%," <<"Removing seams : " << int(float(i)/atoi(argv[2]) *100) << "%" << "\r" <<std::flush;
    }
    video_out.write(frame_result);

  }
  std::cout<<std::endl;
  
  // When everything done, release the video video_inture object
  video_in.release();
  video_out.release();
  // Closes all the frames
  cv::destroyAllWindows();
     
  return 0;
}

/*image storing
std::stringstream gradient_name,result_name;
gradient_name << "./gradient/gradient"<< i << ".png"; 
result_name << "./result/result"<< i << ".png" ;
cv::imwrite( gradient_name.str().c_str(), gradient ); 
cv::imwrite( result_name.str().c_str(), result ); 
*/