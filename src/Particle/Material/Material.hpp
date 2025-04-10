//
// Created by iqraa on 15-2-25.
//

#ifndef MATERIAL_H
#define MATERIAL_H

#include <string>
#include <Tools/Config/Parser.h>
#include <cuda_runtime_api.h>

class Material {
public:

    // Constructors
    __host__ __device__
    Material() = default;

    // Copy constructor const
    __host__ __device__
    Material(const Material& other)
        :materialId_(other.materialId_),
          density_(other.density_),
          youngsModulus_(other.youngsModulus_),
          poissonRatio_(other.poissonRatio_),
          frictionCoeff_(other.frictionCoeff_),
          restitutionCoeff_(other.restitutionCoeff_),
          shearModulus_(other.shearModulus_),
          effectiveYoungsModulus_(other.effectiveYoungsModulus_)
    {}

    // Copy constructor
    __host__ __device__
    Material(Material& other)
        :materialId_(other.materialId_),
          density_(other.density_),
          youngsModulus_(other.youngsModulus_),
          poissonRatio_(other.poissonRatio_),
          frictionCoeff_(other.frictionCoeff_),
          restitutionCoeff_(other.restitutionCoeff_),
          shearModulus_(other.shearModulus_),
          effectiveYoungsModulus_(other.effectiveYoungsModulus_)
    {}

    /// Constructor by variable/file input
    explicit Material(const std::string& path)
    {
        const nlohmann::json data = Parser::readJson(path);

        materialId_ = data["materialId"];
        density_ = data["density"];
        youngsModulus_ = data["youngsModulus"];
        poissonRatio_ = data["poissonRatio"];
        frictionCoeff_ = data["frictionCoeff"];
        restitutionCoeff_ = data["restitutionCoeff"];

        // Calculate derived properties
        calculateShearModulus();
        calculateEffectiveYoungsModulus();
    }

    /// Constructor by variable/file input
    Material(int materialId,
            float density,
            float youngsModulus,
            float poissonRatio,
            float frictionCoeff,
            float restitutionCoeff)
        :materialId_(materialId),
          density_(density),
          youngsModulus_(youngsModulus),
          poissonRatio_(poissonRatio),
          frictionCoeff_(frictionCoeff),
          restitutionCoeff_(restitutionCoeff)
    {
        // Calculate derived properties
        calculateShearModulus();
        calculateEffectiveYoungsModulus();
    }

    __host__ __device__
    virtual ~Material() = default;

    // Getters
    __host__ __device__
    int getMaterialId() const { return materialId_; }

    __host__ __device__
    float getDensity() const { return density_; }

    float getYoungsModulus() const { return youngsModulus_; }

    float getPoissonRatio() const { return poissonRatio_; }

    float getFrictionCoeff() const { return frictionCoeff_; }

    float getRestitutionCoeff() const { return restitutionCoeff_; }

    float getShearModulus() const { return shearModulus_; }

    float getEffectiveYoungsModulus() const { return effectiveYoungsModulus_; }

    // Setters
    void setMaterialId(const int materialId) { materialId_ = materialId; }
    void setDensity(float density) { density_ = density; }

    void setYoungsModulus(float youngsModulus) {
        youngsModulus_ = youngsModulus;
        calculateShearModulus();
        calculateEffectiveYoungsModulus();
    }

    void setPoissonRatio(float poissonRatio) {
        poissonRatio_ = poissonRatio;
        calculateShearModulus();
        calculateEffectiveYoungsModulus();
    }

    void setFrictionCoeff(float frictionCoeff) {
        frictionCoeff_ = frictionCoeff;
    }

    void setRestitutionCoeff(float restitutionCoeff) {
        restitutionCoeff_ = restitutionCoeff;
    }

    // Calculate contact parameters between two materials
    static float calculateEffectiveYoungsModulus(const Material& mat1, const Material& mat2)
    {
        float E1 = mat1.getYoungsModulus();
        float E2 = mat2.getYoungsModulus();
        float v1 = mat1.getPoissonRatio();
        float v2 = mat2.getPoissonRatio();

        return 1.0f / ((1.0f - v1 * v1) / E1 + (1.0f - v2 * v2) / E2);
    }

    static float calculateEffectiveFrictionCoeff(const Material& mat1, const Material& mat2)
    {
        return std::min(mat1.getFrictionCoeff(), mat2.getFrictionCoeff());
    }

    static float calculateEffectiveRestitutionCoeff(const Material& mat1, const Material& mat2) {
        return std::min(mat1.getRestitutionCoeff(), mat2.getRestitutionCoeff());
    }

private:
    // Primary material properties
    int materialId_ = 0;
    float density_ = 1000.0f;            // kg/m³
    float youngsModulus_ = 1.0e7f;       // Pa
    float poissonRatio_  = 0.3f;         // dimensionless
    float frictionCoeff_ = 0.3f;         // dimensionless
    float restitutionCoeff_ = 0.7f;      // dimensionless

    // Derived properties
    float shearModulus_ = 0.0f;          // Pa
    float effectiveYoungsModulus_ = 0.0f;// Pa

    // Helper methods to calculate derived properties
    __host__ __device__
    void calculateShearModulus() {
        shearModulus_ = youngsModulus_ / (2.0f * (1.0f + poissonRatio_));
    }

    __host__ __device__
    void calculateEffectiveYoungsModulus() {
        effectiveYoungsModulus_ = youngsModulus_ / (1.0f - poissonRatio_ * poissonRatio_);
    }
};

#endif //MATERIAL_H
