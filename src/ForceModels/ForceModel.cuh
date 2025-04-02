//
// Created by mbahassan on 4/2/25.
//

#ifndef FORCEMODEL_CUH
#define FORCEMODEL_CUH

enum Model
{
  LSD,
  HertzMindlin
};


class ForceModel
{
  public:
  explicit ForceModel(const Model model = HertzMindlin): model_(model) {};

    private:
      Model model_;
};



#endif //FORCEMODEL_CUH
