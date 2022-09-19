#ifndef DAUGMAN
#define DAUGMAN

#include<iostream>
#include<vector>
#include "opencv2/highgui.hpp"

# define M_PI   3.14159265358979323846

namespace dm {

	/**	Finds radius of cirlce that maximize daugman metric (that is the maximum difference of 
	* normalized sum  of pixel intensisties along the circumference of given consecutive radius)
	* The algorithm searches the best values of delta and its corresponding radius, between 
	* a starting and ending values with step increment. 
	* The value of delta is weigthed according to pixel intensities inside the cirlce.
	* 
	* @param img - Image of the eye
	* @param p - center point to apply algorithm
	* @param start_r - starting value of radius
	* @param end_r - ending value of radius
	* @param step - increment for the radius at each iteration
	* @return vector of double containing weighted delta [0] and radius [1] found.
	* 
	*/
	std::vector<double> daugman(cv::Mat img, cv::Point p, int start_r, int end_r, int step = 1);

	/** Find best iris (center radius) according to daugman algorithm applied to a grid centered
	* in the image with width scale 3/5 and heihgt scale 1/3. Density of points in the grid is 
	* determined according to parameter "p_step". 
	*
	* @param img - Image of the eye
	* @param start_r - daugman starting value of radius
	* @param end_r - daugman ending value of radius
	* @param d_step - daugman increment for the radius
	* @param p_step - step for grid points (the higher the denser the grid)
	* @return vector of int containing coordinates of the center (x[0],y[1]) and radius r[2]
	* of the best iris found.
	* 
	*/
	std::vector<int> findIris(cv::Mat img, int d_start, int d_end, int d_step = 1, int p_step = 1);


	std::vector<int> printIris(cv::Mat src, cv::Mat* dst, cv::Rect r);

}


#endif
