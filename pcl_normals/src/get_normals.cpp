#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>

int main(int argc, char* argv[]) {
  pcl::PointCloud<pcl::PointXYZ> cloud;

  cloud.width = 2088 * 64;
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.points.resize(cloud.width * cloud.height);

	std::string line;
  std::ifstream myfile (argv[1]);
  if (myfile.is_open())
  {
    int ind = 0;
    while (getline(myfile, line))
    {
      char temp[line.length() + 1];
      memset(temp, 0, line.length() + 1);
      memcpy(temp, line.c_str(), line.length());

      cloud.points[ind].x = std::stof(strtok(temp, ",")); 
      cloud.points[ind].y = std::stof(strtok(NULL, ","));
      cloud.points[ind].z = std::stof(strtok(NULL, ","));
      ind++;
      printf("%f %f %f\n", cloud.points[ind].x, cloud.points[ind].y, cloud.points[ind].z); 
    }
    myfile.close();
  } else {
    printf("Could not open file.\n"); 
  }
}
