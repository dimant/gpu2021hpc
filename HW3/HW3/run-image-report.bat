
.\HW3.exe -k imageFilter -i .\Big-Gray-Lena_8bit.png -o gaussianBlur7x7.jpg -f gaussianBlur7x7
.\HW3.exe -k imageFilter -i .\Big-Gray-Lena_8bit.png -o gaussianBlur5x5.jpg -f gaussianBlur5x5
.\HW3.exe -k imageFilter -i .\Big-Gray-Lena_8bit.png -o compositeLaplacian.jpg -f compositeLaplacian
.\HW3.exe -k imageFilter -i .\Big-Gray-Lena_8bit.png -o basicLaplacianDiags.jpg -f basicLaplacianDiags
.\HW3.exe -k imageFilter -i .\Big-Gray-Lena_8bit.png -o sobelEdgeX.jpg -f sobelEdgeX
.\HW3.exe -k imageFilter -i .\Big-Gray-Lena_8bit.png -o identity.jpg -f identity
.\HW3.exe -k imageFilter -i .\Big-Gray-Lena_8bit.png -o average.jpg -f average
.\HW3.exe -k imageFilter -i .\Big-Gray-Lena_8bit.png -o prewittX.jpg -f prewittX
.\HW3.exe -k imageFilter -i .\Big-Gray-Lena_8bit.png -o prewittY.jpg -f prewittY

.\HW3.exe -k imageFilter -i .\Big-Gray-Lena_8bit.png -o sobel-laplacian.jpg -f sobelEdgeX,basicLaplacianDiags
.\HW3.exe -k imageFilter -i .\Big-Gray-Lena_8bit.png -o sobel-gaussian.jpg -f sobelEdgeX,gaussianBlur7x7
.\HW3.exe -k imageFilter -i .\Big-Gray-Lena_8bit.png -o laplacian-gaussian.jpg -f basicLaplacianDiags,gaussianBlur7x7
.\HW3.exe -k imageFilter -i .\Big-Gray-Lena_8bit.png -o sobel-laplacian-gaussian.jpg -f sobelEdgeX,basicLaplacianDiags,gaussianBlur7x7

.\HW3.exe -k t2dPDE_center --reference-impl center --steps 500 --rows 1024 --cols 1024 
.\HW3.exe -k t2dPDE_center --reference-impl center_clamp --steps 500 --rows 1024 --cols 1024 
.\HW3.exe -k t2dPDE_center --reference-impl full --steps 500 --rows 1024 --cols 1024 
.\HW3.exe -k t2dPDE_center_clamp --reference-impl center_clamp --steps 500 --rows 1024 --cols 1024 
.\HW3.exe -k t2dPDE_center_clamp --reference-impl center --steps 500 --rows 1024 --cols 1024 
.\HW3.exe -k t2dPDE_center_clamp --reference-impl full --steps 500 --rows 1024 --cols 1024 
.\HW3.exe -k t2dPDE_full --reference-impl full --steps 500 --rows 1024 --cols 1024 
.\HW3.exe -k t2dPDE_full --reference-impl center --steps 500 --rows 1024 --cols 1024 
.\HW3.exe -k t2dPDE_full --reference-impl center_clamp --steps 500 --rows 1024 --cols 1024 