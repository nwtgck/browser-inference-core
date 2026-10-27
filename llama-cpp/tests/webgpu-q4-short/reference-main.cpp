// The preceding dequantizer and scale helper are extracted unchanged from the
// pinned upstream ggml-quants.c. This wrapper computes a double-precision dot
// from their F32 outputs, independently of shader lane/reduction mapping.
#include <iostream>
#include <stdexcept>
#include <vector>
static void read_bytes(void * data,size_t bytes){if(!std::cin.read(static_cast<char *>(data),bytes))throw std::runtime_error("truncated fixture");}
int main(){try{
    uint32_t h[5];read_bytes(h,sizeof(h));const bool indexed=h[0];
    if(h[0]>1 || h[1]>256*1024*1024 || h[2]>16*1024*1024 || h[3]>1024 || h[4]!=(indexed?13u:16u))throw std::runtime_error("invalid fixture header");
    std::vector<uint8_t> weights(h[1]);std::vector<float> input(h[2]);std::vector<uint32_t> ids(h[3]),p(h[4]);
    read_bytes(weights.data(),weights.size());read_bytes(input.data(),input.size()*4);read_bytes(ids.data(),ids.size()*4);read_bytes(p.data(),p.size()*4);
    const uint32_t k=p[indexed?4:5],m=p[indexed?5:3],slots=indexed?p[7]:p[12]*p[14]*p[13]*p[15];
    if(!k || k%256 || !m || !slots || k>65536 || m>65536 || slots>1024)throw std::runtime_error("invalid dimensions");
    std::vector<float> row(k);std::vector<double> result(size_t(slots)*m*2);
    for(uint32_t slot=0;slot<slots;slot++)for(uint32_t r=0;r<m;r++){
        size_t aoff,xoff;
        if(indexed){if(p[2]+slot>=ids.size() || !p[8])throw std::runtime_error("invalid ids/input count");aoff=p[0]+ids[p[2]+slot]*size_t(p[11])+r*size_t(p[9]);xoff=p[1]+(slot%p[8])*size_t(p[10]);}
        else {const uint32_t i2=slot%(p[12]*p[14]),i3=slot/(p[12]*p[14]);aoff=p[0]+(i3/p[15])*size_t(p[10])+(i2/p[14])*size_t(p[8])+r*size_t(p[6]);xoff=p[1]+i3*size_t(p[11])+i2*size_t(p[9]);}
        if((aoff+k/256)*144>weights.size() || xoff+k>input.size())throw std::runtime_error("reference range exceeded");
        dequantize_row_q4_K(reinterpret_cast<const block_q4_K *>(weights.data()+aoff*144),row.data(),k);
        double sum=0,absolute=0;for(uint32_t i=0;i<k;i++){const double product=double(row[i])*input[xoff+i];sum+=product;absolute+=std::abs(product);}
        const size_t out=(size_t(slot)*m+r)*2;result[out]=sum;result[out+1]=absolute;
    }
    std::cout.write(reinterpret_cast<const char *>(result.data()),result.size()*sizeof(double));
}catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}}
