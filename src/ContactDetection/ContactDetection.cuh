//
// Created by mbahassan on 2/28/25.
//

#ifndef CONTACTDETECTION_CUH
#define CONTACTDETECTION_CUH



class ContactDetection {
    public:
    enum Method {OCTREE, QUADTREE, KDTREE};

    ContactDetection(std::vector<Particle> &particles, Method method);

    void detectContacts();


};



#endif //CONTACTDETECTION_CUH
