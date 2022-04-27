/* -*-c++-*- OpenSceneGraph - Copyright (C) 1998-2010 Robert Osfield
 *
 * This library is open source and may be redistributed and/or modified under
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * OpenSceneGraph Public License for more details.
*/

/* ##################################################################################################### */
/* ParallelSplitShadowMap written by Adrian Egli (3dhelp (at) gmail.com)                                 */
/* ##################################################################################################### */
/*                                                                                                       */
/* the pssm main idea is based on:                                                                       */
/*                                                                                                       */
/* Parallel-Split Shadow Maps for Large-scale Virtual Environments                                       */
/*    Fan Zhang     Hanqiu Sun    Leilei Xu    Lee Kit Lun                                               */
/*    The Chinese University of Hong Kong                                                                */
/*                                                                                                       */
/* Refer to our latest project webpage for "Parallel-Split Shadow Maps on Programmable GPUs" in GPU Gems */
/*                                                                                                       */
/* ##################################################################################################### */

/* ##################################################################################################### */
/* 2022 technique improveed by Marcin Hajder (marcin.hajder (at) gmail.com)                              */
/* ##################################################################################################### */
/*                                                                                                       */
/* instead of _number_of_splits "Camera passes" we can perform one.                                      */
/*                                                                                                       */
/* -Render target is changed to Texture2DArray,                                                          */
/* -for each layer we need matrix uniform which transform vertex from main camera view space...          */
/*  ... to shadow tile post perspective projection.                                                      */
/* -at the shadow casting shader pass additional geometry shader is needed to set gl_Layer.              */
/*                                                                                                       */
/* TODO: adapt technique to use VirtualProgram                                                           */
/* -adapt technique to use VirtualProgram                                                                */
/* -correct main camera view matrix to take into account peripheral shadows.                             */
/*                                                                                                       */
/* ##################################################################################################### */

#ifndef PSSM_LAYERED_H
#define PSSM_LAYERED_H 1

#include <osg/Camera>
#include <osg/Material>
#include <osg/Depth>
#include <osg/ClipPlane>
#include <osg/BufferIndexBinding>
#include <osg/BufferObject>
#include <osg/BufferTemplate>
#include <osg/Texture2DArray>

#include <osgShadow/ShadowTechnique>

namespace osgShadow {

class PSSMLayered :  public ShadowTechnique
{
    public:
        PSSMLayered(osg::Geode** debugGroup=NULL, int icountplanes=3);

        PSSMLayered(const PSSMLayered& es, const osg::CopyOp& copyop=osg::CopyOp::SHALLOW_COPY);

        META_Object(osgShadow, PSSMLayered);


        /** Initialize the ShadowedScene and local cached data structures.*/
        void init() override;

        /** Run the update traversal of the ShadowedScene and update any loca chached data structures.*/
        void update(osg::NodeVisitor& nv) override;

        /** Run the cull traversal of the ShadowedScene and set up the rendering for this ShadowTechnique.*/
        void cull(osgUtil::CullVisitor& cv) override;

        /** Clean scene graph from any shadow technique specific nodes, state and drawables.*/
        void cleanSceneGraph() override;

        /** Switch on the debug coloring in GLSL (only the first 3 texture/splits showed for visualisation */
        inline void setDebugColorOn() { _debug_color_in_GLSL = true; }

        /** Set the polygon offset osg::Vec2f(factor,unit) */
        inline void setPolygonOffset(const osg::Vec2f& p) { _polgyonOffset = p;_user_polgyonOffset_set=true;}

        /** Get the polygon offset osg::Vec2f(factor,unit) */
        inline const osg::Vec2f& getPolygonOffset() const { return _polgyonOffset;}

        /** Set the texture resolution */
        inline void setTextureResolution(unsigned int resolution) { _resolution = resolution; }

        /** Get the texture resolution */
        inline unsigned int getTextureResolution() const { return _resolution; }

        /** Set the max far distance */
        inline void setMaxFarDistance(double farDist) { _setMaxFarDistance = farDist; _isSetMaxFarDistance = true; }

        /** Get the max far distance */
        inline double getMaxFarDistance() const { return _setMaxFarDistance; }

        /** Set the factor for moving the virtual camera behind the real camera*/
        inline void setMoveVCamBehindRCamFactor(double distFactor ) { _move_vcam_behind_rcam_factor = distFactor; }

        /** Get the factor for moving the virtual camera behind the real camera*/
        inline double getMoveVCamBehindRCamFactor() const { return _move_vcam_behind_rcam_factor; }

        /** Set min near distance for splits */
        inline void setMinNearDistanceForSplits(double nd){ _split_min_near_dist=nd; }

        /** Get min near distance for splits */
        inline double getMinNearDistanceForSplits() const { return _split_min_near_dist; }

        /** set a user defined light for shadow simulation (sun light, ... )
         *    when this light get passed to pssm, the scene's light are no longer collected
         *    and simulated. just this user passed light, it needs to be a directional light.
         */
        inline void setUserLight(osg::Light* light) { _userLight = light; }

        /** get the user defined light for shadow simulation */
        inline const osg::Light* getUserLight() const { return _userLight.get(); }

        /** Set the values for the ambient bias the shader will use.*/
        void setAmbientBias(const osg::Vec2& ambientBias );

        /** Get the values for the ambient bias the shader will use.*/
        const osg::Vec2& getAmbientBias() const { return _ambientBias; }

        /** enable / disable shadow filtering */
        inline void enableShadowGLSLFiltering(bool filtering = true) {  _GLSL_shadow_filtered = filtering; }

        enum SplitCalcMode {
            SPLIT_LINEAR,
            SPLIT_EXP
        };

        /** set split calculation mode */
        inline void setSplitCalculationMode(SplitCalcMode scm=SPLIT_EXP) { _SplitCalcMode = scm; }

        /** get split calculation mode */
        inline SplitCalcMode getSplitCalculationMode() const { return _SplitCalcMode; }


        /** Resize any per context GLObject buffers to specified size. */
        void resizeGLObjectBuffers(unsigned int maxSize) override;

        /** If State is non-zero, this function releases any associated OpenGL objects for
           * the specified graphics context. Otherwise, releases OpenGL objects
           * for all graphics contexts. */
        void releaseGLObjects(osg::State* = 0) const override;

    public :

        ~PSSMLayered() override {}

        osg::ref_ptr<osg::Camera>                   _camera;
        osg::ref_ptr<osg::Texture2DArray>           _texture;
        osg::ref_ptr<osg::StateSet>                 _stateset;

        osg::Vec3d                                  _lightDirection;
        osg::Matrixd                                _cameraView;
        osg::Matrixd                                _cameraProj;

        unsigned int                                _textureUnit;

        struct LayerTextureData 
        {
            double                            _split_far;

            // Light (SUN)
            osg::Vec3d                        _lightCameraSource;
            osg::Vec3d                        _lightCameraTarget;
            osg::Vec3d                        _frustumSplitCenter;
            double                            _lightNear;
            double                            _lightFar;

            unsigned int                      _splitID;
            unsigned int                      _resolution;

            osg::Vec3d                        _farDistanceSplit;
            osg::Matrixd                      _view;
            osg::Matrixd                      _proj;

        };

        osg::ref_ptr<osg::Uniform> _projUnif;

        typedef std::map<unsigned int,LayerTextureData> LayerTextureDataMap;
        LayerTextureDataMap _layerTextureDataMap;


        LayerTextureDataMap& getPSSMMap() { return _layerTextureDataMap; }


    private:

        void calculateFrustumCorners(LayerTextureData &pssmShadowSplitTexture,osg::Vec3d *frustumCorners);
        void calculateLightInitialPosition(LayerTextureData &pssmShadowSplitTexture,osg::Vec3d *frustumCorners);
        void calculateLightNearFarFormFrustum(LayerTextureData &pssmShadowSplitTexture,osg::Vec3d *frustumCorners);
        void calculateLightViewProjectionFormFrustum(LayerTextureData &pssmShadowSplitTexture,osg::Vec3d *frustumCorners);

        osg::Geode** _displayTexturesGroupingNode;

        unsigned int _textureUnitOffset;

        unsigned int _number_of_splits;

        bool _debug_color_in_GLSL;

        osg::Vec2 _polgyonOffset;
        bool _user_polgyonOffset_set;

        unsigned int _resolution;

        double _setMaxFarDistance;
        bool _isSetMaxFarDistance;

        double _split_min_near_dist;

        double _move_vcam_behind_rcam_factor;

        osg::ref_ptr<osg::Light> _userLight;

        bool            _GLSL_shadow_filtered;
        SplitCalcMode   _SplitCalcMode;

        osg::Uniform*   _ambientBiasUniform;
        osg::Vec2       _ambientBias;

};
}
#endif
