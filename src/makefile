# MacOS compiler
CC = /Applications/Xcode.app/Contents/Developer/usr/bin/g++
CXX = $(CC)

# OSX include paths for Homebrew
CFLAGS = -Wc++11-extensions -std=c++17 -I/opt/homebrew/Cellar/opencv/4.11.0/include/opencv4 -I/usr/local/include -I../include -I/opt/homebrew/Cellar/onnxruntime/1.20.1_2/include/onnxruntime -DENABLE_PRECOMPILED_HEADERS=OFF
CXXFLAGS = $(CFLAGS)

# Library paths, update to where your OpenCV and ONNX Runtime libraries are stored
LDFLAGS = -L/opt/homebrew/Cellar/opencv/4.11.0/lib
LDFLAGS += -L/opt/homebrew/Cellar/onnxruntime/1.20.1_2/lib
LDFLAGS += -L/opt/homebrew/lib
LDFLAGS += -L/opt/homebrew/opt/protobuf/lib

# Libraries
LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -ljasper -lwebp -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lonnxruntime -lprotobuf

# Output Directory
BINDIR = ../bin

# Source Files
PROGRAMS = displayVid confusionMatrix
SOURCES = displayVid.cpp objectRecPreprocessing.cpp

# Default Target: Build all programs
all: $(PROGRAMS)

# Individual Targets
displayVid: displayVid.o objectRecPreprocessing.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
	install_name_tool -add_rpath /opt/homebrew/lib $(BINDIR)/$@

confusionMatrix: confusionMatrix.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
	install_name_tool -add_rpath /opt/homebrew/lib $(BINDIR)/$@

# Rule to Compile .cpp to .o
%.o: %.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@

# Clean Rule
clean:
	rm -f *.o *~ $(BINDIR)/*
