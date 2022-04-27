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
/* 2022 technique improvement implemented by Marcin Hajder (marcin.hajder (at) gmail.com)                */
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


#include "PSSMLayered.h"

#include <osgShadow/ShadowedScene>
#include <osg/Notify>
#include <osg/ComputeBoundsVisitor>
#include <osg/PolygonOffset>
#include <osg/CullFace>
#include <osg/io_utils>
#include <iostream>
#include <sstream>
#include <osg/Geode>
#include <osg/Geometry>
#include <osgDB/ReadFile>
#include <osg/Texture1D>
#include <osg/Depth>
#include <osg/ShadeModel>
#include <osg/GLObjects>

using namespace osgShadow;

// split scheme
#define TEXTURE_RESOLUTION  1024

#define ZNEAR_MIN_FROM_LIGHT_SOURCE 5.0
#define MOVE_VIRTUAL_CAMERA_BEHIND_REAL_CAMERA_FACTOR 0.0

#ifndef SHADOW_TEXTURE_DEBUG
#define SHADOW_TEXTURE_GLSL
#endif

//////////////////////////////////////////////////////////////////////////
// clamp variables of any type
template<class Type> inline Type Clamp(Type A, Type Min, Type Max) {
    if(A<Min) return Min;
    if(A>Max) return Max;
    return A;
}

#define min(a,b)            (((a) < (b)) ? (a) : (b))
#define max(a,b)            (((a) > (b)) ? (a) : (b))

//////////////////////////////////////////////////////////////////////////
PSSMLayered::PSSMLayered(osg::Geode** gr, int icountplanes) :
    _textureUnitOffset(1),
    _debug_color_in_GLSL(false),
    _user_polgyonOffset_set(false),
    _resolution(TEXTURE_RESOLUTION),
    _setMaxFarDistance(1000.0),
    _isSetMaxFarDistance(false),
    _split_min_near_dist(ZNEAR_MIN_FROM_LIGHT_SOURCE),
    _move_vcam_behind_rcam_factor(MOVE_VIRTUAL_CAMERA_BEHIND_REAL_CAMERA_FACTOR),
    _userLight(NULL),
    _GLSL_shadow_filtered(true),
    _ambientBiasUniform(NULL),
    _ambientBias(0.1f,0.3f)
{
    _displayTexturesGroupingNode = gr;
    _number_of_splits = icountplanes;

    _polgyonOffset.set(0.0f,0.0f);
    setSplitCalculationMode(SPLIT_EXP);
}

PSSMLayered::PSSMLayered(const PSSMLayered& copy, const osg::CopyOp& copyop):
    ShadowTechnique(copy,copyop),
    _displayTexturesGroupingNode(0),
    _textureUnitOffset(copy._textureUnitOffset),
    _number_of_splits(copy._number_of_splits),
    _debug_color_in_GLSL(copy._debug_color_in_GLSL),
    _polgyonOffset(copy._polgyonOffset),
    _user_polgyonOffset_set(copy._user_polgyonOffset_set),
    _resolution(copy._resolution),
    _setMaxFarDistance(copy._setMaxFarDistance),
    _isSetMaxFarDistance(copy._isSetMaxFarDistance),
    _split_min_near_dist(copy._split_min_near_dist),
    _move_vcam_behind_rcam_factor(copy._move_vcam_behind_rcam_factor),
    _userLight(copy._userLight),
    _GLSL_shadow_filtered(copy._GLSL_shadow_filtered),
    _SplitCalcMode(copy._SplitCalcMode),
    _ambientBiasUniform(NULL),
    _ambientBias(copy._ambientBias)
{
}

void PSSMLayered::resizeGLObjectBuffers(unsigned int maxSize)
{
  if(_camera.valid())
    _camera->resizeGLObjectBuffers(maxSize);
  if (_texture.valid())
    _texture->resizeGLObjectBuffers(maxSize);
  if (_stateset.valid())
    _stateset->resizeGLObjectBuffers(maxSize);
}

void PSSMLayered::releaseGLObjects(osg::State* state) const
{
  _camera->releaseGLObjects(state);
  _texture->releaseGLObjects(state);
  _stateset->releaseGLObjects(state);
}


void PSSMLayered::setAmbientBias(const osg::Vec2& ambientBias)
{
    _ambientBias = ambientBias;
    if (_ambientBiasUniform ) _ambientBiasUniform->set(osg::Vec2f(_ambientBias.x(), _ambientBias.y()));
}

void PSSMLayered::init()
{
    if (!_shadowedScene) return;

    unsigned int iCamerasMax=_number_of_splits;

    // create one instance of shared resources:
    //-camera,
    //-TextureArray attached to camera,
    //-uniform array
    // we still need separate LayerTexture's to keep layer data

    // set up the texture to render into
    if(!_texture.valid())
    {
      _texture = new osg::Texture2DArray;
      _texture->setTextureSize(_resolution, _resolution, iCamerasMax);
      _texture->setInternalFormat(GL_DEPTH_COMPONENT24);
      _texture->setSourceFormat(GL_DEPTH_COMPONENT);
      _texture->setSourceType(GL_FLOAT);
      _texture->setShadowComparison(true);
      _texture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
      _texture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
      _texture->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::CLAMP_TO_EDGE);
      _texture->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP_TO_EDGE);
    }

    if (!_projUnif.valid())
    {
      _projUnif = new osg::Uniform(osg::Uniform::FLOAT_MAT4, "projMat", iCamerasMax);
    }

    // set up the render to texture camera.
    if(!_camera.valid())
    {
      // create the camera
      _camera = new osg::Camera;
      _camera->setCullCallback(new CameraCullCallback(this));
      _camera->setClearMask(GL_DEPTH_BUFFER_BIT);
      _camera->setClearDepth(1.0);
      _camera->setComputeNearFarMode(osg::Camera::DO_NOT_COMPUTE_NEAR_FAR);
      _camera->setReferenceFrame(osg::Camera::ABSOLUTE_RF);
      _camera->setImplicitBufferAttachmentMask(0);
      _camera->setViewport(0, 0, _resolution, _resolution);
      _camera->setRenderOrder(osg::Camera::PRE_RENDER);
      _camera->setCullingMode(_camera->getCullingMode() & ~osg::CullStack::SMALL_FEATURE_CULLING);
      _camera->setInheritanceMask(_camera->getInheritanceMask() & ~osg::Camera::CULL_MASK);
      _camera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
      _camera->attach(osg::Camera::DEPTH_BUFFER, _texture.get(), 0, osg::Camera::FACE_CONTROLLED_BY_GEOMETRY_SHADER, false);
      _stateset = new osg::StateSet();
      _stateset->setDataVariance(osg::Object::DYNAMIC);
      _stateset->addUniform(_projUnif);

      //////////////////////////////////////////////////////////////////////////
      if (_user_polgyonOffset_set) 
      {
        float factor = _polgyonOffset.x();
        float units = _polgyonOffset.y();
        osg::ref_ptr<osg::PolygonOffset> polygon_offset = new osg::PolygonOffset;
        polygon_offset->setFactor(factor);
        polygon_offset->setUnits(units);
        _stateset->setAttribute(polygon_offset.get(), osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
        _stateset->setMode(GL_POLYGON_OFFSET_FILL, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
      }

      //////////////////////////////////////////////////////////////////////////
      if (!_GLSL_shadow_filtered) 
      {
        // if not glsl filtering enabled then we should force front face culling to reduce the number of shadow artifacts.
        osg::ref_ptr<osg::CullFace> cull_face = new osg::CullFace;
        cull_face->setMode(osg::CullFace::FRONT);
        _stateset->setAttribute(cull_face.get(), osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
        _stateset->setMode(GL_CULL_FACE, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
      }

      //////////////////////////////////////////////////////////////////////////
      osg::ShadeModel* shademodel = dynamic_cast<osg::ShadeModel*>(_stateset->getAttribute(osg::StateAttribute::SHADEMODEL));
      if (!shademodel) 
      { 
        shademodel = new osg::ShadeModel; _stateset->setAttribute(shademodel);
      }
      shademodel->setMode(osg::ShadeModel::FLAT);
      _stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    }

    for (unsigned int iCameras = 0; iCameras < iCamerasMax; iCameras++)
    {
      LayerTextureData pssmShadowSplitTexture;
      pssmShadowSplitTexture._splitID = iCameras;
      pssmShadowSplitTexture._resolution = _resolution;

      _layerTextureDataMap.insert(LayerTextureDataMap::value_type(iCameras, pssmShadowSplitTexture));
    }

    _dirty = false;
}

//////////////////////////////////////////////////////////////////////////

void PSSMLayered::update(osg::NodeVisitor& nv)
{
    getShadowedScene()->osg::Group::traverse(nv);
}

//////////////////////////////////////////////////////////////////////////

void PSSMLayered::cull(osgUtil::CullVisitor& cv)
{
    // record the traversal mask on entry so we can reapply it later.
    unsigned int traversalMask = cv.getTraversalMask();
    osgUtil::RenderStage* orig_rs = cv.getRenderStage();

    // do traversal of shadow receiving scene which does need to be decorated by the shadow map
    {
      cv.setTraversalMask(traversalMask & getShadowedScene()->getReceivesShadowTraversalMask());

      _shadowedScene->osg::Group::traverse(cv);
    }

    //////////////////////////////////////////////////////////////////////////
    const osg::Light* selectLight = 0;

    /// light pos and light direction
    osg::Vec4d lightpos;
    osg::Vec3d lightDirection;

    if ( ! _userLight ) 
    {
        // try to find a light in the scene
        osgUtil::PositionalStateContainer::AttrMatrixList& aml = orig_rs->getPositionalStateContainer()->getAttrMatrixList();
        for(osgUtil::PositionalStateContainer::AttrMatrixList::iterator itr = aml.begin();
            itr != aml.end();
            ++itr)
        {
            const osg::Light* light = dynamic_cast<const osg::Light*>(itr->first.get());
            if (light)
            {
                osg::RefMatrix* matrix = itr->second.get();
                if (matrix) lightpos = osg::Vec4d(light->getPosition()) * (*matrix);
                else lightpos = light->getPosition();
                if (matrix) lightDirection = osg::Vec3d(light->getDirection()) * (*matrix);
                else lightDirection = light->getDirection();

                selectLight = light;
            }
        }

        osg::Matrixd eyeToWorld;
        eyeToWorld.invert(*cv.getModelViewMatrix());

        lightpos = lightpos * eyeToWorld;
        lightDirection = lightDirection * eyeToWorld;
    }
    else
    {
        // take the user light as light source
        lightpos = _userLight->getPosition();
        lightDirection = _userLight->getDirection();
        selectLight = _userLight.get();
    }

    if (selectLight)
    {
      cv.computeNearPlane();

      _cameraProj = *cv.getProjectionMatrix();

      if (cv.getComputeNearFarMode() && cv.getCalculatedFarPlane() >= cv.getCalculatedNearPlane())
      {
        osgUtil::CullVisitor::value_type znear = cv.getCalculatedNearPlane();
        osgUtil::CullVisitor::value_type zfar = cv.getCalculatedFarPlane();

        cv.clampProjectionMatrixImplementation(_cameraProj, znear, zfar);
      }



      //////////////////////////////////////////////////////////////////////////
      // SETUP pssmShadowSplitTexture for rendering
      //
      lightDirection.normalize();
      _lightDirection = lightDirection;
      _cameraView = cv.getRenderInfo().getView()->getCamera()->getViewMatrix();

      _camera->setViewMatrix(_cameraView);
      _camera->setProjectionMatrix(_cameraProj);

      for(LayerTextureDataMap::iterator it=_layerTextureDataMap.begin();it!=_layerTextureDataMap.end();it++)
      {
          LayerTextureData & pssmShadowSplitTexture = it->second;

          osg::Vec3d pCorners[8];
          calculateFrustumCorners(pssmShadowSplitTexture,pCorners);
          calculateLightInitialPosition(pssmShadowSplitTexture,pCorners);
          calculateLightNearFarFormFrustum(pssmShadowSplitTexture,pCorners);
          calculateLightViewProjectionFormFrustum(pssmShadowSplitTexture,pCorners);
      }

      // do RTT camera traversal
      {
        cv.setTraversalMask(getShadowedScene()->getCastsShadowTraversalMask());

        cv.pushStateSet(_stateset);

        _camera->accept(cv);

        cv.popStateSet();
      }
    } // if light

    // reapply the original traversal mask
    cv.setTraversalMask( traversalMask );
}

void PSSMLayered::cleanSceneGraph()
{

}

//////////////////////////////////////////////////////////////////////////
// Computes corner points of a frustum
//
//
//unit box representing frustum in clip space
const osg::Vec3d const_pointFarTR(   1.0,  1.0,  1.0);
const osg::Vec3d const_pointFarBR(   1.0, -1.0,  1.0);
const osg::Vec3d const_pointFarTL(  -1.0,  1.0,  1.0);
const osg::Vec3d const_pointFarBL(  -1.0, -1.0,  1.0);
const osg::Vec3d const_pointNearTR(  1.0,  1.0, -1.0);
const osg::Vec3d const_pointNearBR(  1.0, -1.0, -1.0);
const osg::Vec3d const_pointNearTL( -1.0,  1.0, -1.0);
const osg::Vec3d const_pointNearBL( -1.0, -1.0, -1.0);
//////////////////////////////////////////////////////////////////////////


void PSSMLayered::calculateFrustumCorners(LayerTextureData &pssmShadowSplitTexture, osg::Vec3d *frustumCorners)
{
    // get user cameras
    double fovy,aspectRatio,camNear,camFar;
    _cameraProj.getPerspective(fovy,aspectRatio,camNear,camFar);


    // force to max far distance to show shadow, for some scene it can be solve performance problems.
    if ((_isSetMaxFarDistance) && (_setMaxFarDistance < camFar))
        camFar = _setMaxFarDistance;


    // build camera matrix with some offsets (the user view camera)
    osg::Matrixd viewMat;
    osg::Vec3d camEye,camCenter,camUp;
    _cameraView.getLookAt(camEye,camCenter,camUp);
    osg::Vec3d viewDir = camCenter - camEye;
    //viewDir.normalize(); //we can assume that viewDir is still normalized in the viewMatrix
    camEye = camEye  - viewDir * _move_vcam_behind_rcam_factor;
    camFar += _move_vcam_behind_rcam_factor * viewDir.length();
    viewMat.makeLookAt(camEye,camCenter,camUp);



    //////////////////////////////////////////////////////////////////////////
    /// CALCULATE SPLIT
    double maxFar = camFar;
    // double minNear = camNear;
    double camNearFar_Dist = maxFar - camNear;
    if ( _SplitCalcMode == SPLIT_LINEAR )
    {
        camFar  = camNear + (camNearFar_Dist) * ((double)(pssmShadowSplitTexture._splitID+1))/((double)(_number_of_splits));
        camNear = camNear + (camNearFar_Dist) * ((double)(pssmShadowSplitTexture._splitID))/((double)(_number_of_splits));
    }
    else
    {
        // Exponential split scheme:
        //
        // Ci = (n - f)*(i/numsplits)^(bias+1) + n;
        //
        static double fSplitSchemeBias[2]={0.25f,0.66f};
        fSplitSchemeBias[1]=Clamp(fSplitSchemeBias[1],0.0,3.0);
        double* pSplitDistances =new double[_number_of_splits+1];

        for(int i=0;i<(int)_number_of_splits;i++)
        {
            double fIDM=(double)(i)/(double)(_number_of_splits);
            pSplitDistances[i]=camNearFar_Dist*(pow(fIDM,fSplitSchemeBias[1]+1))+camNear;
        }
        // make sure border values are right
        pSplitDistances[0]=camNear;
        pSplitDistances[_number_of_splits]=camFar;

        camNear = pSplitDistances[pssmShadowSplitTexture._splitID];
        camFar  = pSplitDistances[pssmShadowSplitTexture._splitID+1];

        delete[] pSplitDistances;
    }

    pssmShadowSplitTexture._split_far = camFar;

    //////////////////////////////////////////////////////////////////////////
    /// TRANSFORM frustum corners (Optimized for Orthogonal)


    osg::Matrixd projMat;
    projMat.makePerspective(fovy,aspectRatio,camNear,camFar);
    osg::Matrixd projViewMat(viewMat*projMat);
    osg::Matrixd invProjViewMat;
    invProjViewMat.invert(projViewMat);

    //transform frustum vertices to world space
    frustumCorners[0] = const_pointFarBR * invProjViewMat;
    frustumCorners[1] = const_pointNearBR* invProjViewMat;
    frustumCorners[2] = const_pointNearTR* invProjViewMat;
    frustumCorners[3] = const_pointFarTR * invProjViewMat;
    frustumCorners[4] = const_pointFarTL * invProjViewMat;
    frustumCorners[5] = const_pointFarBL * invProjViewMat;
    frustumCorners[6] = const_pointNearBL* invProjViewMat;
    frustumCorners[7] = const_pointNearTL* invProjViewMat;
}

//////////////////////////////////////////////////////////////////////////
//
// compute directional light initial position;
void PSSMLayered::calculateLightInitialPosition(LayerTextureData &pssmShadowSplitTexture,osg::Vec3d *frustumCorners)
{
    pssmShadowSplitTexture._frustumSplitCenter = frustumCorners[0];
    for(int i=1;i<8;i++)
    {
        pssmShadowSplitTexture._frustumSplitCenter +=frustumCorners[i];
    }
    //    pssmShadowSplitTexture._frustumSplitCenter /= 8.0;
    pssmShadowSplitTexture._frustumSplitCenter *= 0.125;
}

void PSSMLayered::calculateLightNearFarFormFrustum(
    LayerTextureData &pssmShadowSplitTexture,
    osg::Vec3d *frustumCorners
    ) {

        //calculate near, far
        double zFar(-DBL_MAX);

        // calculate zFar (as longest distance)
        for(int i=0;i<8;i++) {
            double dist_z_from_light = fabs(_lightDirection*(frustumCorners[i] -  pssmShadowSplitTexture._frustumSplitCenter));
            if ( zFar  < dist_z_from_light ) zFar  = dist_z_from_light;
        }

        // update camera position and look at center
        pssmShadowSplitTexture._lightCameraSource = pssmShadowSplitTexture._frustumSplitCenter - _lightDirection*(zFar+_split_min_near_dist);
        pssmShadowSplitTexture._lightCameraTarget = pssmShadowSplitTexture._frustumSplitCenter + _lightDirection*(zFar);

        // calculate [zNear,zFar]
        zFar = (-DBL_MAX);
        double zNear(DBL_MAX);
        for(int i=0;i<8;i++) {
            double dist_z_from_light = fabs(_lightDirection*(frustumCorners[i] -  pssmShadowSplitTexture._lightCameraSource));
            if ( zFar  < dist_z_from_light ) zFar  = dist_z_from_light;
            if ( zNear > dist_z_from_light ) zNear  = dist_z_from_light;
        }
        // update near - far plane
        pssmShadowSplitTexture._lightNear = max(zNear - _split_min_near_dist - 0.01,0.01);
        pssmShadowSplitTexture._lightFar  = zFar;
}

void PSSMLayered::calculateLightViewProjectionFormFrustum(LayerTextureData &pssmShadowSplitTexture,osg::Vec3d *frustumCorners)
{
    // calculate the camera's coordinate system
    osg::Vec3d camEye,camCenter,camUp;
    _cameraView.getLookAt(camEye,camCenter,camUp);
    osg::Vec3d viewDir(camCenter-camEye);
    osg::Vec3d camRight(viewDir^camUp);

    // we force to have normalized vectors (camera's view)
    camUp.normalize();
    viewDir.normalize();
    camRight.normalize();

    // use quaternion -> numerical more robust
    osg::Quat qRot;
    qRot.makeRotate(viewDir, _lightDirection);
    osg::Vec3d top =  qRot * camUp;
    osg::Vec3d right = qRot * camRight;

    // calculate the camera's frustum right,right,bottom,top parameters
    double maxRight(-DBL_MAX),maxTop(-DBL_MAX);
    double minRight(DBL_MAX),minTop(DBL_MAX);

    for(int i(0); i < 8; i++)
    {

        osg::Vec3d diffCorner(frustumCorners[i] - pssmShadowSplitTexture._frustumSplitCenter);
        double lright(diffCorner*right);
        double lTop(diffCorner*top);

        if ( lright > maxRight ) maxRight  =  lright;
        if ( lTop  > maxTop  ) maxTop   =  lTop;

        if ( lright < minRight ) minRight  =  lright;
        if ( lTop  < minTop  ) minTop   =  lTop;
    }

    pssmShadowSplitTexture._view = osg::Matrixd::lookAt
      (pssmShadowSplitTexture._lightCameraSource, pssmShadowSplitTexture._lightCameraTarget, top);

    pssmShadowSplitTexture._proj = osg::Matrixd::ortho
      (minRight, maxRight, minTop, maxTop, pssmShadowSplitTexture._lightNear, pssmShadowSplitTexture._lightFar);
    _projUnif->setElement(pssmShadowSplitTexture._splitID, 
      osg::Matrixf(osg::Matrix::inverse(_cameraView) * pssmShadowSplitTexture._view * pssmShadowSplitTexture._proj));

    osg::Vec3d vProjCamFraValue = (camEye + viewDir * pssmShadowSplitTexture._split_far) * (_cameraView * _cameraProj);
    pssmShadowSplitTexture._farDistanceSplit = vProjCamFraValue;
}



