#include <iostream>
#include "lodepng.h"
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>

using namespace std;

#define SAFE_CALL(CallInstruction) {\
    cudaError_t cuerr = CallInstruction; \
    if (cuerr != cudaSuccess) { \
        printf("CUDA error: %s at call \"" #CallInstruction "\"\n", cudaGetErrorString(cuerr)); \
            throw "error in CUDA API function, aborting..."; \
    } \
}

#define SAFE_KERNEL_CALL(KernelCallInstruction) { \
    KernelCallInstruction; \
    cudaError_t cuerr = cudaGetLastError(); \
    if (cuerr != cudaSuccess) { \
        printf("CUDA error in kernel launch: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
            throw "error in CUDA kernel launch, aborting..."; \
    } \
    cuerr = cudaDeviceSynchronize(); \
    if (cuerr != cudaSuccess) { \
        printf("CUDA error in kernel execution: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
            throw "error in CUDA kernel execution, aborting..."; \
    } \
}

__global__ void GaussianBlur5x5(uchar3 *inp, uchar3 *out,
                                   int width, int height, int padding) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_size = width - padding * 2;
    if ((x < width - 2) && (y < height - 2)
        && (x > 1) && (y > 1)) {
        int idx = y * width + x;
        int coeff = 2 * width + 2 + ((idx - 2 * width - 2 - 4 * (y - 2)) / out_size) * 2 * padding;
        /*
                [ 1  4  6  4 1
                  4 16 24 16 4
        1/256 *   6 24 36 24 6
                  4 16 24 16 4
                  1  4  6  4 1]
        */
        double new_x, new_y, new_z;
        new_x = inp[idx - 2 * width - 2].x / 256. + 4 * inp[idx - 2 * width - 1].x / 256.
                + 6 * inp[idx - 2 * width].x / 256. + 4 * inp[idx - 2 * width + 1].x / 256.
                + inp[idx - 2 * width + 1].x / 256. + 4 * inp[idx - width - 2].x / 256.
                + 16 * inp[idx - width - 1].x / 256. + 24 * inp[idx - width].x / 256.
                + 16 * inp[idx - width + 1].x / 256. + 4 * inp[idx - width + 2].x / 256.
                + 6 * inp[idx - 2].x / 256. + 24 * inp[idx - 1].x / 256.
                + 36 * inp[idx].x / 256. + 24 * inp[idx + 1].x / 256.
                + 6 * inp[idx + 2].x / 256. + 4 * inp[idx + width - 2].x / 256.
                + 16 * inp[idx + width - 1].x / 256. + 24 * inp[idx + width].x / 256.
                + 16 * inp[idx + width + 1].x / 256. + 4 * inp[idx + width + 2].x / 256.
                + inp[idx + 2 * width - 2].x / 256. + 4 * inp[idx + 2 * width - 1].x / 256.
                + 6 * inp[idx + 2 * width].x / 256. + 4 * inp[idx + 2 * width + 1].x / 256.
                + inp[idx + 2 * width + 1].x / 256.;
        new_y = inp[idx - 2 * width - 2].y / 256. + 4 * inp[idx - 2 * width - 1].y / 256.
                + 6 * inp[idx - 2 * width].y / 256. + 4 * inp[idx - 2 * width + 1].y / 256.
                + inp[idx - 2 * width + 1].y / 256. + 4 * inp[idx - width - 2].y / 256.
                + 16 * inp[idx - width - 1].y / 256. + 24 * inp[idx - width].y / 256.
                + 16 * inp[idx - width + 1].y / 256. + 4 * inp[idx - width + 2].y / 256.
                + 6 * inp[idx - 2].y / 256. + 24 * inp[idx - 1].y / 256.
                + 36 * inp[idx].y / 256. + 24 * inp[idx + 1].y / 256.
                + 6 * inp[idx + 2].y / 256. + 4 * inp[idx + width - 2].y / 256.
                + 16 * inp[idx + width - 1].y / 256. + 24 * inp[idx + width].y / 256.
                + 16 * inp[idx + width + 1].y / 256. + 4 * inp[idx + width + 2].y / 256.
                + inp[idx + 2 * width - 2].y / 256. + 4 * inp[idx + 2 * width - 1].y / 256.
                + 6 * inp[idx + 2 * width].y / 256. + 4 * inp[idx + 2 * width + 1].y / 256.
                + inp[idx + 2 * width + 1].y / 256.;
        new_z = inp[idx - 2 * width - 2].z / 256. + 4 * inp[idx - 2 * width - 1].z / 256.
                + 6 * inp[idx - 2 * width].z / 256. + 4 * inp[idx - 2 * width + 1].z / 256.
                + inp[idx - 2 * width + 1].z / 256. + 4 * inp[idx - width - 2].z / 256.
                + 16 * inp[idx - width - 1].z / 256. + 24 * inp[idx - width].z / 256.
                + 16 * inp[idx - width + 1].z / 256. + 4 * inp[idx - width + 2].z / 256.
                + 6 * inp[idx - 2].z / 256. + 24 * inp[idx - 1].z / 256.
                + 36 * inp[idx].z / 256. + 24 * inp[idx + 1].z / 256.
                + 6 * inp[idx + 2].z / 256. + 4 * inp[idx + width - 2].z / 256.
                + 16 * inp[idx + width - 1].z / 256. + 24 * inp[idx + width].z / 256.
                + 16 * inp[idx + width + 1].z / 256. + 4 * inp[idx + width + 2].z / 256.
                + inp[idx + 2 * width - 2].z / 256. + 4 * inp[idx + 2 * width - 1].z / 256.
                + 6 * inp[idx + 2 * width].z / 256. + 4 * inp[idx + 2 * width + 1].z / 256.
                + inp[idx + 2 * width + 1].z / 256.;
        out[idx - coeff].x = min(255., max(0., new_x));
        out[idx - coeff].y = min(255., max(0., new_y));
        out[idx - coeff].z = min(255., max(0., new_z));
    }
}

__global__ void EdgeDetection3x3_1(uchar3 *inp, uchar3 *out,
                                   int width, int height, int padding) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_size = width - padding * 2;
    if ((x < width - 1) && (y < height - 1)
        && (x > 0) && (y > 0)) {
        int idx = y * width + x;
        int coeff = width + 1 + ((idx - width - 1 - 2 * (y - 1)) / out_size) * 2 * padding;

        /*
            [ 1 0 -1
              0 0  0
             -1 0  1]
        */
        out[idx - coeff].x = inp[idx - width - 1].x - inp[idx - width + 1].x
                        - inp[idx + width - 1].x + inp[idx + width + 1].x;
        out[idx - coeff].y = inp[idx - width - 1].y - inp[idx - width + 1].y
                        - inp[idx + width - 1].y + inp[idx + width + 1].y;
        out[idx - coeff].z = inp[idx - width - 1].z - inp[idx - width + 1].z
                        - inp[idx + width - 1].z + inp[idx + width + 1].z;
    }
}

__global__ void EdgeDetection3x3_2(uchar3 *inp, uchar3 *out,
                                   int width, int height, int padding) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_size = width - padding * 2;
    if ((x < width - 1) && (y < height - 1)
        && (x > 0) && (y > 0)) {
        int idx = y * width + x;
        int coeff = width + 1 + ((idx - width - 1 - 2 * (y - 1)) / out_size) * 2 * padding;
        /*
            [ 0 -1  0
             -1  4 -1
              0 -1  0]
        */
        out[idx - coeff].x = - inp[idx - width ].x - inp[idx - 1].x
                        + 4 * inp[idx].x - inp[idx + 1].x
                        - inp[idx + width].x;
        out[idx - coeff].y = - inp[idx - width ].y - inp[idx - 1].y
                        + 4 * inp[idx].y - inp[idx + 1].y
                        - inp[idx + width].y;
        out[idx - coeff].z = - inp[idx - width ].z - inp[idx - 1].z
                        + 4 * inp[idx].z - inp[idx + 1].z
                        - inp[idx + width].z;
    }
}

pair<unsigned, unsigned> ReadImages(vector<unsigned char *> &images, string images_path,
                int images_count) {
    unsigned width, height;
    for (int i = 0; i < images_count; ++i) {
        stringstream temp;
        temp << i;
        unsigned char *cur_img;
        string file_name = images_path + temp.str() + ".png";
        //decode
        if (lodepng_decode24_file(
            &cur_img, &width, &height, file_name.data()
           ))
            cout << "decoder error " << endl;
        //if there's an error, display it

        images.push_back(cur_img);
    }
    return make_pair(width, height);
}

int main(int argc, char **argv) {
    vector<unsigned char *> images;
    int filter_type = atoi(argv[1]), images_type = atoi(argv[2]),
        input_size, output_size, filter_pad;
    uchar3 *device_input, *device_output;
    unsigned width, height;
    string images_path = images_type == 1 ? "./inputs/300x300/" : "./inputs/2000x2000/",
        outputs_path = "./outputs/";
    pair<unsigned, unsigned> sizes;
    if (images_type == 1)
        sizes = ReadImages(images, images_path, 50);
    else
        sizes = ReadImages(images, images_path, 5);

    width = sizes.first;
    height = sizes.second;


    input_size = width * height * 3 * sizeof(unsigned char);
    SAFE_CALL(cudaMalloc(&device_input, input_size));
    SAFE_CALL(cudaMalloc(&device_output, input_size));

    filter_pad = filter_type == 1 ? 2 : 1;
    output_size = (width - filter_pad * 2) * (height - filter_pad * 2)
                * 3 * sizeof(unsigned char);

    cudaEvent_t full_start, full_stop, kernel_start, kernel_stop;
    float full_time = 0, kernel_time = 0;
    cudaEventCreate(&full_start);
    cudaEventCreate(&full_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    for (int i = 0; i < images.size(); ++i) {
        stringstream temp;
        temp << i;
        unsigned char *cur_out = new unsigned char[output_size];
        //Copying from CPU to GPU
		cudaEventRecord(full_start);
        SAFE_CALL(cudaMemcpy(device_input, images[i],
            input_size, cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMemcpy(device_output, cur_out,
            output_size, cudaMemcpyHostToDevice));

        //Computation
        size_t blockSize = 1,
            gridSize = width;
        dim3 grid(gridSize, gridSize);
		cudaEventRecord(kernel_start);
        if (filter_type == 1) {
            SAFE_KERNEL_CALL((GaussianBlur5x5<<<grid, blockSize>>>(
                device_input, device_output, width, height, filter_pad
            )));
        }
        else if (filter_type == 2) {
            SAFE_KERNEL_CALL((EdgeDetection3x3_1<<<grid, blockSize>>>(
                device_input, device_output, width, height, filter_pad
            )));
        }
        else {
            SAFE_KERNEL_CALL((EdgeDetection3x3_2<<<grid, blockSize>>>(
                device_input, device_output, width, height, filter_pad
            )));
        }
        SAFE_CALL(cudaDeviceSynchronize());
		cudaEventRecord(kernel_stop);

        //Copying from GPU to CPU
        SAFE_CALL(cudaMemcpy(cur_out, device_output,
            output_size, cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaDeviceSynchronize());
		cudaEventRecord(full_stop);

        string file_name = outputs_path + temp.str() + ".png";
        unsigned error = lodepng_encode24_file(file_name.data(),
                                         cur_out, width - filter_pad * 2, height - filter_pad * 2);
        if (error)
            cout << "encoder error " << error << ": "<< lodepng_error_text(error) << endl;

        delete [] cur_out;
		cudaEventSynchronize(full_stop);
		cudaEventSynchronize(kernel_stop);
        float cur_full_time, cur_kernel_time;
		cudaEventElapsedTime(&cur_full_time, full_start, full_stop);
		cudaEventElapsedTime(&cur_kernel_time, kernel_start, kernel_stop);
		cout << "Current kernel time = " << cur_kernel_time << " ms" << endl
             << "Current full time = " << cur_full_time << " ms" << endl;
		full_time += cur_full_time;
		kernel_time += cur_kernel_time;
    }
    cout << "Average kernel time = " << kernel_time / images.size() << " ms" << endl
         << "Average full time = " << full_time / images.size() << " ms" << endl;
    SAFE_CALL(cudaFree(device_input));
    SAFE_CALL(cudaFree(device_output));
    return 0;
}
