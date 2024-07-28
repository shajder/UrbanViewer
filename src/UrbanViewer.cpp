#include <osgDB/ReadFile>
#include <osgDB/WriteFile>

#include <osg/Switch>
#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/FrontFace>
#include <osg/StateSet>
#include <osg/BlendFunc>

#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>

#include <osgGA/TrackballManipulator>
#include <osgGA/FlightManipulator>
#include <osgGA/DriveManipulator>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osgGA/StateSetManipulator>
#include <osgGA/AnimationPathManipulator>
#include <osgGA/TerrainManipulator>
#include <osgGA/SphericalManipulator>

#include <osgShadow/ShadowedScene>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

#include "PSSMLayered.h"
#include "RenderingResources.h"
#include "Shaders.h"


////////////////////////////////////////////////////////////////////////////////

// simple structure to hold global variables in one package
RenderingResources rr;

////////////////////////////////////////////////////////////////////////////////
// specialization of shadow technique to set specific settings for the sake of this demo
namespace osgShadow {

  class PSSMLayeredWrapper : public PSSMLayered
  {
  public:
    using BaseClass = PSSMLayered;

    PSSMLayeredWrapper() {}

    PSSMLayeredWrapper(
      const PSSMLayeredWrapper& copy,
      const osg::CopyOp& copyop = osg::CopyOp::SHALLOW_COPY) : BaseClass(copy, copyop) {}

    PSSMLayeredWrapper(int numMaps) : BaseClass(nullptr, numMaps) {}

    LayerTextureDataMap& getPSSMMap() { return _layerTextureDataMap; }

    void init() override;
    void cull(osgUtil::CullVisitor& cv) override;

  protected:

    META_Object(osgShadow, PSSMLayeredWrapper);
  };
}

////////////////////////////////////////////////////////////////////////////////

using namespace osgShadow;

////////////////////////////////////////////////////////////////////////////////

void PSSMLayeredWrapper::init()
{
  BaseClass::init();

  _stateset->removeAttribute(osg::StateAttribute::PROGRAM);
  _camera->getOrCreateStateSet()->removeAttribute(osg::StateAttribute::PROGRAM);

  const int shadow_tex_stage = 4;

  _camera->setCullMask(SHADOW_MASK);
  _camera->setRenderOrder(osg::Camera::PRE_RENDER, 0);

  auto globalLightStates = rr.sceneRoot->getOrCreateStateSet();
  globalLightStates->setTextureAttributeAndModes(shadow_tex_stage, _texture,
    osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
  globalLightStates->getOrCreateUniform("shadow_tex", osg::Uniform::SAMPLER_2D_ARRAY_SHADOW)->set(shadow_tex_stage);
}

////////////////////////////////////////////////////////////////////////////////

void PSSMLayeredWrapper::cull(osgUtil::CullVisitor& cv)
{
  if (_layerTextureDataMap.empty()) return;
  
  BaseClass::cull(cv);
}

////////////////////////////////////////////////////////////////////////////////
// Uniforms could be updated right after culling traversal is done.
// At this point bounding boxes were computed and we can calculate near/far.
struct UpdateUniformsCallback : public osg::NodeCallback
{
  UpdateUniformsCallback(){}

  void operator()(osg::Node* node, osg::NodeVisitor* nv) override
  {
    if (nv->getVisitorType() == osg::NodeVisitor::CULL_VISITOR)
    {
      traverse(node, nv);

      // now it is safe to update uniforms
      osgUtil::CullVisitor* cv = nv->asCullVisitor();
      if (!cv) return;

      auto camera = cv->getRenderInfo().getView()->getCamera();

      if (!camera->getViewport() || !camera->getViewport()->valid())
        return;

      if (cv->getRenderInfo().getView() && cv->getRenderInfo().getView()->getCamera())
      {
        osg::Matrixd viewMat = camera->getViewMatrix();
        osg::Matrixd projMat = camera->getProjectionMatrix();

        if (cv->getComputeNearFarMode() && cv->getCalculatedFarPlane() >= cv->getCalculatedNearPlane())
        {
          cv->computeNearPlane();
          osgUtil::CullVisitor::value_type znear = cv->getCalculatedNearPlane();
          osgUtil::CullVisitor::value_type zfar = cv->getCalculatedFarPlane();

          cv->clampProjectionMatrixImplementation(projMat, znear, zfar);
        }

        {
          if (rr.sunLight.valid())
          {
            osg::Vec3d lpos = rr.sunPos;
            rr.sunLight->setPosition(osg::Vec4(lpos[0], lpos[1], lpos[2], 0));
            lpos.normalize();
            rr.sunLight->setDirection(-lpos);
          }

          {
            auto vp = camera->getViewport();
            auto viewportUnif = rr.sceneRoot->getOrCreateStateSet()->getOrCreateUniform
            ("viewport", osg::Uniform::FLOAT_VEC2);
            viewportUnif->set(osg::Vec2(float(vp->width()), float(vp->height())));
          }

          {
            osg::Vec3d ldir;
            auto lightDirBase = rr.sunPos;

            ldir = osg::Matrix::transform3x3(lightDirBase, viewMat);
            ldir.normalize();

            auto lightBaseDirUnif = rr.sceneRoot->getOrCreateStateSet()->getOrCreateUniform
            ("lightPos", osg::Uniform::FLOAT_VEC3);
            lightBaseDirUnif->set(osg::Vec3f(ldir));
          }
        }

        // first update non-uniform stuff
        if (rr.pssm.valid())
        {

          {
            auto pssm_map = rr.pssm->getPSSMMap();
            auto globalLightStates = rr.sceneRoot->getOrCreateStateSet();

            int counter = 0;

            for (auto& it : pssm_map)
            {
              osg::Matrix shadowView, shadowProj;

              shadowView = it.second._view;
              shadowProj = it.second._proj;

              std::stringstream sstr;
              sstr << "zShadow" << counter;
              globalLightStates->getOrCreateUniform(sstr.str(), osg::Uniform::FLOAT)->set
              (float(it.second._farDistanceSplit.z()));

              {
                osg::Matrixf mat =
                  osg::Matrix::inverse(viewMat) *
                  shadowView *
                  shadowProj *
                  osg::Matrix::translate(1.0, 1.0, 1.0) *
                  osg::Matrix::scale(0.5f, 0.5f, 0.5f);


                std::stringstream sstr;
                sstr.str(std::string());
                sstr << "tex_mat" << counter;
                globalLightStates->getOrCreateUniform(sstr.str(), osg::Uniform::FLOAT_MAT4)->set(mat);
              }

              counter++;
            }
          }
        }

        // update AO uniform buffer values
        if (rr.hbaoBuf.valid())
        {
          const double* P = projMat.ptr();
          rr.hbaoBuf->getData().projInfo.set(
            2.0f / (P[4 * 0 + 0]),                  // (x) * (R - L)/N
            2.0f / (P[4 * 1 + 1]),                  // (y) * (T - B)/N
            -(1.0f - P[4 * 2 + 0]) / P[4 * 0 + 0],  // L/N
            -(1.0f + P[4 * 2 + 1]) / P[4 * 1 + 1]   // B/N
          );

          auto viewportUnif = rr.sceneRoot->getOrCreateStateSet()->getOrCreateUniform("viewport", osg::Uniform::FLOAT_VEC2);
          osg::Vec2 vp;
          viewportUnif->get(vp);

          double fovy, ar, zn, zf;
          projMat.getPerspective(fovy, ar, zn, zf);
          if (zf == 0.0)
            fovy = 30.f;
          fovy = osg::DegreesToRadians(fovy);
          double projScale = double(vp.y()) / (tanf(fovy * 0.5f) * 2.0f);

          // radius
          float meters2viewspace = 1.0f;
          float R = rr.hbaoSetup.radius * meters2viewspace;
          rr.hbaoBuf->getData().rad2 = R * R;
          rr.hbaoBuf->getData().negInvRad2 = -1.0f / rr.hbaoBuf->getData().rad2;
          rr.hbaoBuf->getData().radToScreen = R * 0.5f * projScale;

          if (rr.clipInfoUnif.valid())
          {
            rr.clipInfoUnif->set(osg::Vec4f(zn, zn - zf, zf, 1.f));
          }

          rr.hbaoBuf->dirty();
        }
      }

      return;
    }
    traverse(node, nv);
  }
};

////////////////////////////////////////////////////////////////////////////////
// Class to adapt shaders for PSSM shadowing
// _programs_fixed contains fixed shaders for receiving pass
// _programs contains original shaders for casting pass
class AdaptProg2PSSMVisitor : public osg::NodeVisitor
{
  std::string _frag_pattern;
  std::string _vert_pattern;

  std::set<osg::ref_ptr<osg::Program> > _programs;
  std::set<osg::ref_ptr<osg::Program> > _programs_fixed;

  osg::ref_ptr<osg::Shader> _vshader;
  osg::ref_ptr<osg::Shader> _fshader;

  //////////////////////////////////////////////////////////////////////////////

public:

  std::set<osg::ref_ptr<osg::Program> >& getProgramsSet() { return _programs; }
  std::set<osg::ref_ptr<osg::Program> >& getPSSMAdaptedProgramsSet() { return _programs_fixed; }

  //////////////////////////////////////////////////////////////////////////////

  AdaptProg2PSSMVisitor(osg::Shader* vshader, osg::Shader* fshader, const std::string&& vpat, const std::string&& fpat) :
    _vshader(vshader),
    _fshader(fshader),
    _vert_pattern(vpat),
    _frag_pattern(fpat),
    osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
  {
  }

  //////////////////////////////////////////////////////////////////////////////

  void apply(osg::Node& node) override
  {
    apply(node.getStateSet());
    traverse(node);
  }

  //////////////////////////////////////////////////////////////////////////////

  void apply(osg::StateSet* pss)
  {
    if (!pss) return;
    osg::Program* prog = dynamic_cast<osg::Program*>(pss->getAttribute(osg::StateAttribute::PROGRAM));

    if (prog)
    {
      osg::ref_ptr<osg::Program> fixed = (osg::Program*)prog->clone(osg::CopyOp::DEEP_COPY_ALL);

      bool replace = false;
      osg::Shader* frag = nullptr;
      osg::Shader* vert = nullptr;
      for (unsigned i = 0; i < fixed->getNumShaders(); i++)
      {
        osg::Shader* shader = fixed->getShader(i);
        switch (shader->getType())
        {
        case osg::Shader::FRAGMENT:
          frag = shader;
          break;
        case osg::Shader::VERTEX:
          vert = shader;
          break;
        }
      }

      if (frag && !_frag_pattern.empty())
      {
        std::string source = frag->getShaderSource();
        std::string nsource = source;

        unsigned sl = _frag_pattern.size();
        auto pos = source.find(_frag_pattern);

        if (pos < source.size())
        {
          nsource =
            source.substr(0, pos) + _fshader->getShaderSource() + source.substr(pos + sl);

          std::stringstream sstrb;
#ifdef _ANDROID_
          sstrb << "precision mediump float; \n";
#endif                                                                                

          // adapt shaders for multi-layered shadow cast processing
          sstrb << " uniform sampler2DArrayShadow shadow_tex;\n";
          for (int i = 0; i < rr.numLayers; i++)
          {
            sstrb << " uniform mat4 tex_mat" << i << ";\n";
            sstrb << " uniform float zShadow" << i << "; " << std::endl;
          }

          nsource = sstrb.str() + nsource;

          auto inj_pos = nsource.find("return ocol;");
          if (inj_pos < nsource.size())
          {
            std::stringstream sstre;

            sstre << "vec4 tc0; \n";
            sstre << "float testZ = gl_FragCoord.z*2.0-1.0;" << std::endl;
            sstre << "float map0 = step(testZ, zShadow0);" << std::endl;
            for (int i = 1; i < rr.numLayers; i++)
            {
              sstre << "float map" << i << "  = step(zShadow" << i - 1 << ",testZ)*step(testZ, zShadow" << i << ");" << std::endl;
            }

            float step = 1.f / rr.numLayers;

            for (int i = 0; i < rr.numLayers; i++)
            {
              sstre << " tc0 = tex_mat" << i << " * ecp; \n";
              sstre << " tc0 = vec4(tc0.x, tc0.y, " << std::fixed << std::setprecision(1) << double(i) << ", tc0.z);\n";
              sstre << " float shadow" << i << " = texture( shadow_tex, tc0 ); \n";
              sstre << " shadow" << i << " = step(" << step << ",shadow" << i << "); \n";
            }

            sstre << "    float term0 = (1.0-shadow0)*map0; " << std::endl;
            for (int i = 1; i < rr.numLayers; i++)
            {
              sstre << "    float term" << i << " = map" << i << "*(1.0-shadow" << i << ");" << std::endl;
            }

            sstre << "    float v = clamp(";
            for (int i = 0; i < rr.numLayers; i++) {
              sstre << "term" << i;
              if (i + 1 < rr.numLayers) {
                sstre << "+";
              }
            }
            sstre << ",0.0,1.0);" << std::endl;

            sstre << "    ocol.rgb *= vec3(mix(1.0, 0.5, v)); " << std::endl;
            sstre << "    ocol.a = 1.0; \n";
            sstre << "    return ocol; \n";

            nsource = nsource.substr(0, inj_pos) + sstre.str() + nsource.substr(inj_pos + 12);
          }

          // adapt shaders for multi-layered shadow cast processing
          if (nsource.find("#version") == std::string::npos)
            nsource =
            "#version 430 compatibility \n"
            + nsource;
        }
        else
        {
          // report problem
          std::cout << "Shader has no ColorFilter method !!!" << std::endl;
        }

        replace = true;
        frag->setShaderSource(nsource);
        frag->dirtyShader();
      }


      if (vert && !_vert_pattern.empty())
      {
        std::string source = vert->getShaderSource();

        unsigned sl = _vert_pattern.size();
        unsigned pos = source.find(_vert_pattern);
        std::string nsource = source;

        if (pos < source.size())
        {
          nsource =
            source.substr(0, pos) +
            _vshader->getShaderSource() +
            source.substr(pos + sl);
        }
        else
        {
          // report problem
          std::cout << "Shader has no filter: " << _vert_pattern << std::endl;
        }

        replace = true;
        vert->setShaderSource(nsource);
        vert->dirtyShader();
      }

      if (replace)
      {
        _programs.insert(prog);
        _programs_fixed.insert(fixed);
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Class to adapt shaders for PSSM shadow casting. The idea is ...
// to add geometry shader which will select specific layer for FBO target.
// _programs contains added geometry shaders.
class AdaptCastProg2PSSMVisitor : public osg::NodeVisitor
{
  std::set<osg::ref_ptr<osg::Program> > _programs;

public:

  std::set<osg::ref_ptr<osg::Program> >& getProgramsSet() { return _programs; }

  AdaptCastProg2PSSMVisitor() :
    osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
  {
  }

  //////////////////////////////////////////////////////////////////////////////
  void apply(osg::Node& node) override
  {
    apply(node.getStateSet());
    traverse(node);
  }

  //////////////////////////////////////////////////////////////////////////////
  void apply(osg::StateSet* pss)
  {
    if (!pss) return;
    osg::Program* prog = dynamic_cast<osg::Program*>(pss->getAttribute(osg::StateAttribute::PROGRAM));

    bool replace = false;
    if (prog)
    {
      // correct default geometry shader
      osg::Shader* geom = new osg::Shader(osg::Shader::GEOMETRY, layered_shadow_cast_geom);
      if (geom)
      {
        std::string source = geom->getShaderSource();
        std::string nsource = source;
        std::string pattern = "void main()";


        unsigned sl = pattern.size();
        auto pos = source.find(pattern);

        if (pos < source.size())
        {
          std::stringstream sstr;
          sstr << "#define NUM_LAYERS " << rr.numLayers << std::endl;
          sstr << "uniform mat4 projMat[NUM_LAYERS];\n";

          nsource =
            source.substr(0, pos) +
            sstr.str() +
            source.substr(pos);

          replace = true;
          geom->setShaderSource(nsource);
          geom->dirtyShader();
        }

        prog->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, 3 * rr.numLayers);
        prog->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, GL_TRIANGLES);
        prog->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
        prog->addShader(geom);
      }

      if (replace)
      {
        _programs.insert(prog);
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ComputeBoundsVisitor : public osg::NodeVisitor
{
  ComputeBoundsVisitor(osg::BoundingBox& bb) : _bb(bb),
    osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
  {
  }

  void apply(osg::Geode& geo) override
  {
    osg::Matrixf matrix = osg::computeLocalToWorld(this->getNodePath());
    for (unsigned j = 0; j < geo.getNumDrawables(); ++j) 
    {
      osg::Geometry* geom = geo.getDrawable(j)->asGeometry();
      if (!geom) continue;
      osg::Vec3Array* verts = (osg::Vec3Array*)geom->getVertexArray();
      for (unsigned k = 0; k < verts->size(); k++)
        _bb.expandBy(matrix.preMult((*verts)[k]));
    }
  }

  osg::BoundingBox& _bb;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void setupTiledScene(osg::Node* node)
{
  auto createTex = [](const std::string&& name)
  {
    osg::Texture2D* ptex = new osg::Texture2D(osgDB::readImageFile(name));
    ptex->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR);
    ptex->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
    ptex->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
    ptex->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
    ptex->setUseHardwareMipMapGeneration(true);
    return ptex;
  };

  if (!rr.culture2.valid())
  {
    rr.culture2 = createTex("data/sidewalk.tga");
  }

  node->getOrCreateStateSet()->setTextureAttributeAndModes(1, rr.culture2,
    osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE );

  if (!rr.culture1.valid())
  {
    rr.culture1 = createTex("data/grass.tga");
  }

  node->getOrCreateStateSet()->setTextureAttributeAndModes(2, rr.culture1,
    osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE );

  if (!rr.culture3.valid())
  {
    rr.culture3 = createTex("data/pnoise0.tga");
  }

  node->getOrCreateStateSet()->setTextureAttribute(3, rr.culture3, 
    osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE );
  node->getOrCreateStateSet()->addUniform(new osg::Uniform("sampler3", 3));
  node->getOrCreateStateSet()->addUniform(new osg::Uniform("lightAmbient", osg::Vec4(0.5f, 0.5f, 0.5f, 0.5f)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

osg::Group* preProcessCity(osg::Group* bld, osg::Group* ter)
{
  if (!bld || !ter)
    return nullptr;

  osg::Group* city = new osg::Group();

  bld = bld->getChild(0)->asGroup();    // buildings
  ter = ter->getChild(0)->asGroup();    // tiles

  setupTiledScene(ter);

  osg::Group* depth_parent = new osg::Group;
  osg::Group* forward_parent = new osg::Group;

  {
    forward_parent->getOrCreateStateSet()->setAttribute
      (new osg::FrontFace(osg::FrontFace::COUNTER_CLOCKWISE), osg::StateAttribute::ON);
    depth_parent->getOrCreateStateSet()->setAttribute
      (new osg::FrontFace(osg::FrontFace::COUNTER_CLOCKWISE), osg::StateAttribute::ON);
    forward_parent->addChild(bld);
    depth_parent->addChild(bld);

    depth_parent->setNodeMask(SHADOW_MASK); // depth scene with simple shader
    forward_parent->setNodeMask(~SHADOW_MASK); // forward scene with regular shader

    bld = new osg::Group;
    bld->addChild(forward_parent);
    bld->addChild(depth_parent);
  }

  {
    ter->setNodeMask(~SHADOW_MASK); // tiles cast no shadows
    city->addChild(bld);
    city->addChild(ter);
  }

  {
    // shadow receiving buildings, later to be replaced by VirtualProgram
    // for now visitor works as shader compositor which will replace given patterns.
    osg::ref_ptr<osg::Shader > vshader;
    osg::ref_ptr<osg::Shader > fshader;
    vshader = new osg::Shader(osg::Shader::VERTEX, terrain_color_filter_vert);
    fshader = new osg::Shader(osg::Shader::FRAGMENT, terrain_color_filter_PSSM_frag);
    std::string vpat = "vec4 vert_proc(in vec4 vert, in vec3 ecnor){return vert;}";
    std::string fpat = "vec4 ColorFilter( in vec4 color ){return color;}";

    // first visitor, it will not modify original program
    AdaptProg2PSSMVisitor rvpv(vshader, fshader, std::move(vpat), std::move(fpat));
    forward_parent->accept(rvpv);

    // second visitor, it will modifiy original program to adapt for layered casting pass
    AdaptCastProg2PSSMVisitor rcpv;
    forward_parent->accept(rcpv);

    if (!rvpv.getProgramsSet().empty())
    {
      if (rvpv.getProgramsSet().size() > 1)
      {
        std::cerr << "Error: Assuming specific structure of data file" << std::endl;
      }
      else
      {
        auto program = *rcpv.getProgramsSet().begin();
        auto fixed = *rvpv.getPSSMAdaptedProgramsSet().begin();
        for (unsigned i = 0; i < program->getNumParents(); i++)
          program->getParent(i)->removeAttribute(program);

        forward_parent->getOrCreateStateSet()->setAttribute(fixed);
        depth_parent->getOrCreateStateSet()->setAttribute(program);
      }
    }
  }

  // correct shaders for shadow receiving tiles
  {
    osg::ref_ptr<osg::Shader > vshader;
    osg::ref_ptr<osg::Shader > fshader;
    vshader = new osg::Shader(osg::Shader::VERTEX, terrain_filter_vert);
    fshader = new osg::Shader(osg::Shader::FRAGMENT, tiled_terrain_color_filter_PSSM_frag);
    std::string vpat = "vec4 vert_proc(in vec4 vert, in vec3 ecnor){return vert;}";
    std::string fpat = "vec4 ColorFilter( in vec4 color ){return color;}";

    AdaptProg2PSSMVisitor rvpv(vshader, fshader, std::move(vpat), std::move(fpat));
    rvpv.setTraversalMask(~FADE_MASK);
    ter->accept(rvpv);

    if (!rvpv.getProgramsSet().empty())
    {
      for (auto& it : rvpv.getProgramsSet())
      {
        auto program = it;
        for (unsigned i = 0; i < program->getNumParents(); i++)
          program->getParent(i)->removeAttribute(program);
      }

      auto fixed = *rvpv.getPSSMAdaptedProgramsSet().begin();
      ter->getOrCreateStateSet()->setAttribute(fixed);
    }
  }
  return city;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

osg::Group* createShadowScene(std::string& buildings, std::string& terrain, osgViewer::Viewer* viewer)
{
  osg::Node* bld = osgDB::readNodeFile(buildings);
  if (!bld) return NULL;

  osg::Node* ter = osgDB::readNodeFile(terrain);
  if (!ter) return NULL;

  auto city = preProcessCity(bld->asGroup(), ter->asGroup());

  const float sunDist = 1000000.f;
  osg::Vec3 sunDir = osg::Z_AXIS * 0.4f +
                     osg::Y_AXIS * 0.5f +
                     osg::X_AXIS * 0.4f;

  sunDir.normalize();
  rr.sunPos = sunDir * sunDist;

  rr.sunLight = new osg::Light();
  rr.sunLight->setLightNum(0);
  rr.sunLight->setPosition(osg::Vec4(rr.sunPos[0], rr.sunPos[1], rr.sunPos[2], 0));

  auto scene = new osg::Group();

  osg::DisplaySettings::instance()->setImplicitBufferAttachmentRenderMask(0);

  auto lightDirUnif = new osg::Uniform("lightPos", osg::Z_AXIS);
  scene->getOrCreateStateSet()->addUniform(lightDirUnif);

  auto lightDifUnif = new osg::Uniform("lightDiffuse", osg::Vec4(0.5f, 0.5f, 0.5f, 1.f));
  scene->getOrCreateStateSet()->addUniform(lightDifUnif);

  auto lightAmbUnif = new osg::Uniform("lightAmbient", osg::Vec4(0.5f, 0.5f, 0.5f, 1.f));
  scene->getOrCreateStateSet()->addUniform(lightAmbUnif);

  scene->getOrCreateStateSet()->addUniform(new osg::Uniform("sampler0", 0));
  scene->getOrCreateStateSet()->addUniform(new osg::Uniform("sampler1", 1));
  scene->getOrCreateStateSet()->addUniform(new osg::Uniform("sampler2", 2));

  {
    osgViewer::Viewer::Windows windows;
    viewer->getWindows(windows);
    for (auto& itr : windows)
    {
      itr->getState()->setUseModelViewAndProjectionUniforms(true);
      itr->getState()->setUseVertexAttributeAliasing(true);
    }
  }

#if USE_SHADOW
  auto shadowedScene = new osgShadow::ShadowedScene;

  float maxFarPlane = 100000.f;
  unsigned int texSize = 4096;
  float minNearSplit = 10000.f;
  float moveVCamFactor = 0.f;

  auto pssm = new osgShadow::PSSMLayeredWrapper(rr.numLayers);

  pssm->setTextureResolution(texSize);
  pssm->setMaxFarDistance(maxFarPlane);
  pssm->setPolygonOffset(osg::Vec2(1.1f, 4.f));
  pssm->setMinNearDistanceForSplits(minNearSplit);
  pssm->setMoveVCamBehindRCamFactor(moveVCamFactor);
  pssm->setUserLight(rr.sunLight);

  shadowedScene->setReceivesShadowTraversalMask(~SHADOW_MASK);
  shadowedScene->setCastsShadowTraversalMask(SHADOW_MASK);
  shadowedScene->setShadowTechnique(pssm);

  scene->addChild(shadowedScene);
  shadowedScene->addChild(city);

  rr.pssm = pssm;

#else

  scene->addChild(city);

#endif

  scene->setCullCallback(new UpdateUniformsCallback());

  rr.sceneRoot = scene;

  return scene;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void setupHBOAPass(int width, int height)
{
  rr.hbaoSetup.samples = 1;
  rr.hbaoSetup.intensity = 1.f;
  rr.hbaoSetup.bias = 0.f;
  rr.hbaoSetup.radius = float(width) / 24;
  rr.hbaoSetup.blurSharpness = 40.0f;
  rr.hbaoSetup.blur = false;
  rr.hbaoSetup.ortho = false;

  if (!rr.hbaoBuf.valid())
    rr.hbaoBuf = new osg::BufferTemplate< RenderingResources::HBAOData >;

  // ao
  rr.hbaoBuf->getData().powExp = std::max(rr.hbaoSetup.intensity, 0.0f);
  rr.hbaoBuf->getData().NDotVBias = std::min(std::max(0.0f, rr.hbaoSetup.bias), 1.0f);
  rr.hbaoBuf->getData().AOMult = 1.0f / (1.0f - rr.hbaoBuf->getData().NDotVBias);

  // resolution
  int quarterWidth = ((width + 3) / 4);
  int quarterHeight = ((height + 3) / 4);

  rr.ssao_down_w = quarterWidth;
  rr.ssao_down_h = quarterHeight;

  rr.hbaoBuf->getData().invQuarterRes = osg::Vec2(1.0f / float(quarterWidth), 1.0f / float(quarterHeight));
  rr.hbaoBuf->getData().invFullRes = osg::Vec2(1.0f / float(width), 1.0f / float(height));

  if (!rr.hbaoUBO.valid())
  {
    rr.hbaoUBO = new osg::UniformBufferObject();
    rr.hbaoBuf->setBufferObject(rr.hbaoUBO.get());
    rr.hbaoUBB = new osg::UniformBufferBinding(0, rr.hbaoBuf.get(), 0, rr.hbaoBuf->getTotalDataSize());

    std::mt19937 rmt;
    float numDir = 8;  // keep in sync to glsl

    for (int i = 0; i < HBAO_RANDOM_ELEMENTS * MAX_SAMPLES; i++)
    {
      float Rand1 = static_cast<float>(rmt()) / 4294967296.0f;
      float Rand2 = static_cast<float>(rmt()) / 4294967296.0f;

      // Use random rotation angles in [0,2PI/NUM_DIRECTIONS)
      float Angle = 2.f * osg::PI * Rand1 / numDir;

      rr.hbaoRandom[i][0] = Angle;
      rr.hbaoRandom[i][1] = Rand2;
    }

    for (int i = 0; i < HBAO_RANDOM_ELEMENTS; i++)
    {
      rr.hbaoBuf->getData().jitters[i] = rr.hbaoRandom[i];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

osg::Projection* createScreenQuad (int w, int h, int numInstaces)
{
  osg::Geode* geode = new osg::Geode();
  geode->setCullingActive(false);

  osg::Geometry* hud = new osg::Geometry;
  hud->setUseDisplayList(false);
  hud->setUseVertexBufferObjects(true);

  osg::Vec3Array* vertices = new osg::Vec3Array;
  float depth = -0.1;
  vertices->asVector() =
  {
    osg::Vec3(0, h, depth),
    osg::Vec3(0, 0, depth),
    osg::Vec3(w, 0, depth),
    osg::Vec3(w, 0, depth),
    osg::Vec3(w, h, depth),
    osg::Vec3(0, h, depth)
  };

  hud->setVertexArray(vertices);

  osg::Vec2Array* texCoords = new osg::Vec2Array;
  texCoords->asVector() = { osg::Vec2(0, 1), osg::Vec2(0, 0), osg::Vec2(1, 0),
    osg::Vec2(1, 0), osg::Vec2(1, 1), osg::Vec2(0, 1) };
  hud->setTexCoordArray(0, texCoords);

  auto prims = new osg::DrawArrays(GL_TRIANGLES, 0, 6);
  prims->setNumInstances(numInstaces);
  hud->addPrimitiveSet(prims);

  geode->addDrawable(hud);

  osg::MatrixTransform* modelview_abs = new osg::MatrixTransform;
  modelview_abs->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
  modelview_abs->setMatrix(osg::Matrixf::identity());
  modelview_abs->addChild(geode);

  osg::Projection* projection = new osg::Projection;
  projection->setMatrix(osg::Matrixf::ortho2D(0, w, 0, h));
  projection->addChild(modelview_abs);
  return projection;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void readDisplay(unsigned & x, unsigned & y)
{
  osg::GraphicsContext::WindowingSystemInterface* wsi = osg::GraphicsContext::getWindowingSystemInterface();
  if (!wsi)
  {
    osg::notify(osg::NOTICE) << "Error, no WindowSystemInterface available, cannot create windows." << std::endl;
    return;
  }

  osg::GraphicsContext::ScreenIdentifier main_screen_id;

  main_screen_id.readDISPLAY();
  main_screen_id.setUndefinedScreenDetailsToDefaultScreen();
  wsi->getScreenResolution(main_screen_id, x, y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

osg::Group* createScreenCamera(osg::Group* scene, unsigned x, unsigned y)
{
  osg::ref_ptr<osg::Group> root = new osg::Group;

  // setup color target texture
  osg::ref_ptr<osg::Texture2D> colorTarget = new osg::Texture2D;
  colorTarget->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
  colorTarget->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
  colorTarget->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
  colorTarget->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
  colorTarget->setInternalFormat(GL_RGBA);
  colorTarget->setSourceFormat(GL_RGBA);
  colorTarget->setSourceType(GL_UNSIGNED_BYTE);
  colorTarget->setUseHardwareMipMapGeneration(false);
  colorTarget->setResizeNonPowerOfTwoHint(false);

  // setup depth target texture
  osg::Texture2D* depthTarget = new osg::Texture2D();
  depthTarget->setInternalFormat(GL_DEPTH_COMPONENT24);
  depthTarget->setSourceFormat(GL_DEPTH_COMPONENT);
  depthTarget->setSourceType(GL_FLOAT);
  depthTarget->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
  depthTarget->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
  depthTarget->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::CLAMP_TO_EDGE);
  depthTarget->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP_TO_EDGE);
  depthTarget->setUseHardwareMipMapGeneration(false);
  depthTarget->setResizeNonPowerOfTwoHint(false);
  depthTarget->setShadowComparison(false);

  // setup FBO camera, second in order after shadow RTT pass
  auto screenCamera = new osg::Camera();
  screenCamera->setDataVariance(osg::Object::DYNAMIC);
  screenCamera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  screenCamera->setClearColor(osg::Vec4(0.f, 0.f, 0.f, 0.f));
  screenCamera->setClearDepth(1.0);
  screenCamera->setCullingMode(screenCamera->getCullingMode() & ~osg::CullStack::SMALL_FEATURE_CULLING);
  screenCamera->setInheritanceMask(screenCamera->getInheritanceMask() & ~osg::Camera::CULL_MASK);
  screenCamera->setRenderOrder(osg::Camera::PRE_RENDER, 1);
  screenCamera->setReferenceFrame(osg::Camera::RELATIVE_RF);
  screenCamera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
  screenCamera->setViewport(0, 0, x, y);
  screenCamera->setAllowEventFocus(false);

  // specify multisampled FBO
  screenCamera->attach(osg::Camera::COLOR_BUFFER, colorTarget, 0, 0, false, 4, 4);
  screenCamera->attach(osg::Camera::DEPTH_BUFFER, depthTarget, 0, 0, false, 4, 4);
  screenCamera->addChild(scene);
  root->addChild(screenCamera);

  // setup screen space quad to perform deferred pass
  auto projection = createScreenQuad(x, y, 0);
  auto screenRect = new osg::Group;
  root->addChild(screenRect);

  {
    screenRect->addChild(projection);
    osg::StateSet* pss = screenRect->getOrCreateStateSet();
    pss->setTextureAttributeAndModes(0, colorTarget);
    pss->setTextureAttributeAndModes(1, depthTarget);

    {
      osg::Shader* vshader = new osg::Shader(osg::Shader::VERTEX, screen_vert);
      osg::Shader* fshader = new osg::Shader(osg::Shader::FRAGMENT, screen_frag);
      osg::Program* screenProg = new osg::Program;
      screenProg->addShader(vshader);
      screenProg->addShader(fshader);
      pss->setAttribute(screenProg);
    }

    pss->setMode(GL_BLEND, osg::StateAttribute::OFF );
    pss->setMode(GL_CULL_FACE, osg::StateAttribute::ON );
    pss->setAttributeAndModes(new osg::Depth(osg::Depth::ALWAYS, 0, 1, 1), osg::StateAttribute::ON);
    pss->setRenderBinDetails(0, "RenderBin");
    pss->setNestRenderBins(false);
  }

  // setup ambient occlusion pass which will be drawn at the top of screen space quad
  setupHBOAPass(x, y);

  auto create_new_camera = [](float x, float y, osg::Group* root)
  {
    osg::ref_ptr<osg::Camera> camera = new osg::Camera;
    camera->setDataVariance(osg::Object::DYNAMIC);
    camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    camera->setClearColor(osg::Vec4(0.f, 0.f, 0.f, 0.f));
    camera->setClearDepth(1.0);
    camera->setCullingMode(camera->getCullingMode() & ~osg::CullStack::SMALL_FEATURE_CULLING);
    camera->setInheritanceMask(camera->getInheritanceMask() & ~osg::Camera::CULL_MASK);
    camera->setRenderOrder(osg::Camera::PRE_RENDER, 2); // after scene  
    camera->setReferenceFrame(osg::Camera::RELATIVE_RF);
    camera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
    camera->setViewport(0, 0, x, y);
    camera->setAllowEventFocus(false);
    root->addChild(camera);
    return camera.release();
  };

  auto screenRect_linearize = new osg::Group;
  auto screenRect_viewnormal = new osg::Group;
  auto screenRect_hbao2_calc = new osg::Group;
  auto screenRect_reinterleave = new osg::Group;

#if USE_HBAO
  // depth linearize pass
  {
    osg::ref_ptr<osg::Camera> lin_camera = create_new_camera(x, y, root);
    lin_camera->addChild(screenRect_linearize);

    auto screenRect_linearize_tex = new osg::Texture2D();
    screenRect_linearize_tex->setInternalFormat(GL_R32F);
    screenRect_linearize_tex->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::NEAREST);
    screenRect_linearize_tex->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::NEAREST);
    screenRect_linearize_tex->setWrap(osg::Texture2D::WRAP_S, osg::Texture2D::CLAMP_TO_EDGE);
    screenRect_linearize_tex->setWrap(osg::Texture2D::WRAP_T, osg::Texture2D::CLAMP_TO_EDGE);
    screenRect_linearize_tex->setUseHardwareMipMapGeneration(false);
    screenRect_linearize_tex->setResizeNonPowerOfTwoHint(false);
    lin_camera->attach(osg::Camera::COLOR_BUFFER, screenRect_linearize_tex);
    lin_camera->setImplicitBufferAttachmentMask(0);

    screenRect_linearize->addChild(projection);
    osg::StateSet* pss = screenRect_linearize->getOrCreateStateSet();
    pss->setTextureAttribute(0, depthTarget);

    {
      osg::Shader* vshader = new osg::Shader(osg::Shader::VERTEX, screen_vert);
      osg::Shader* fshader = new osg::Shader(osg::Shader::FRAGMENT, screen_linearize_frag);
      osg::Program* screenProg = new osg::Program;
      screenProg->addShader(vshader);
      screenProg->addShader(fshader);
      pss->setAttribute(screenProg);
    }

    {
      rr.clipInfoUnif = new osg::Uniform("clipInfo", osg::Vec4f());
      pss->addUniform(rr.clipInfoUnif);
    }


    // viewnormal pass
    osg::ref_ptr<osg::Camera> vnor_camera = create_new_camera(x, y, root);
    vnor_camera->addChild(screenRect_viewnormal);

    auto screenRect_viewnormal_tex = new osg::Texture2D();
    screenRect_viewnormal_tex->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR);
    screenRect_viewnormal_tex->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
    screenRect_viewnormal_tex->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
    screenRect_viewnormal_tex->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
    screenRect_viewnormal_tex->setInternalFormat(GL_R8);
    screenRect_viewnormal_tex->setSourceFormat(GL_R);
    screenRect_viewnormal_tex->setSourceType(GL_UNSIGNED_BYTE);
    screenRect_viewnormal_tex->setUseHardwareMipMapGeneration(true);
    screenRect_viewnormal_tex->setResizeNonPowerOfTwoHint(false);
    screenRect_viewnormal_tex->setTextureSize(x, y);
    vnor_camera->attach(osg::Camera::COLOR_BUFFER, screenRect_viewnormal_tex);
    vnor_camera->setImplicitBufferAttachmentMask(0);

    {
      screenRect_viewnormal->addChild(projection);
      osg::StateSet* pss = screenRect_viewnormal->getOrCreateStateSet();
      pss->setTextureAttribute(0, screenRect_linearize_tex);
      pss->setAttributeAndModes(rr.hbaoUBB, osg::StateAttribute::ON);

      {
        osg::Shader* vshader = new osg::Shader(osg::Shader::VERTEX, screen_vert);
        osg::Shader* fshader = new osg::Shader(osg::Shader::FRAGMENT, screen_HBAO_frag);
        osg::Program* screenProg = new osg::Program;
        screenProg->addShader(vshader);
        screenProg->addShader(fshader);
        screenProg->addBindUniformBlock("controlBuffer", 0);
        pss->setAttribute(screenProg);
      }
    }

    // scene shade pass
    osg::ref_ptr<osg::Camera> reinterleave_camera = new osg::Camera;
    reinterleave_camera->setDataVariance(osg::Object::DYNAMIC);
    reinterleave_camera->setRenderOrder(osg::Camera::NESTED_RENDER); // after scene  
    reinterleave_camera->setReferenceFrame(osg::Camera::RELATIVE_RF);
    reinterleave_camera->setAllowEventFocus(false);
    reinterleave_camera->addChild(screenRect_reinterleave);
    screenRect->addChild(reinterleave_camera);

    screenRect_reinterleave->addChild(projection);
    {
      osg::StateSet* pss = screenRect_reinterleave->getOrCreateStateSet();
      pss->setTextureAttribute(0, screenRect_viewnormal_tex);
      pss->setAttributeAndModes(new osg::Depth(osg::Depth::ALWAYS, 0, 1, 0), osg::StateAttribute::ON );
      pss->setAttributeAndModes(new osg::BlendFunc(GL_ZERO, GL_SRC_COLOR), osg::StateAttribute::ON );
      pss->setRenderBinDetails(100, "RenderBin");
      pss->setNestRenderBins(false);

      {
        osg::Shader* vshader = new osg::Shader(osg::Shader::VERTEX, screen_vert);
        osg::Shader* fshader = new osg::Shader(osg::Shader::FRAGMENT, screen_apply_SSAO_frag);
        osg::Program* screenProg = new osg::Program;
        screenProg->addShader(vshader);
        screenProg->addShader(fshader);
        pss->setAttribute(screenProg);
      }
    }
  }
#endif

  return root.release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  // use an ArgumentParser object to manage the program arguments.
  osg::ArgumentParser arguments(&argc, argv);

  arguments.getApplicationUsage()->setApplicationName(arguments.getApplicationName());
  arguments.getApplicationUsage()->setDescription(arguments.getApplicationName() + " is the standard OpenSceneGraph example which loads and visualises 3d models.");
  arguments.getApplicationUsage()->setCommandLineUsage(arguments.getApplicationName() + " [options] filename ...");
  arguments.getApplicationUsage()->addCommandLineOption("--image <filename>", "Load an image and render it on a quad");
  arguments.getApplicationUsage()->addCommandLineOption("--dem <filename>", "Load an image/DEM and render it on a HeightField");
  arguments.getApplicationUsage()->addCommandLineOption("--login <url> <username> <password>", "Provide authentication information for http file access.");

  unsigned width = 1920, height = 1080;
#if !HARDCODED_WINDOW_SIZE
  osgViewer::Viewer viewer(arguments);
#else
  osgViewer::Viewer viewer;
  viewer.setUpViewInWindow(0, 0, width, height);
#endif

  unsigned int helpType = 0;
  if ((helpType = arguments.readHelpType()))
  {
    arguments.getApplicationUsage()->write(std::cout, helpType);
    return 1;
  }

  // report any errors if they have occurred when parsing the program arguments.
  if (arguments.errors())
  {
    arguments.writeErrorMessages(std::cout);
    return 1;
  }

  if (arguments.argc() <= 1)
  {
    arguments.getApplicationUsage()->write(std::cout, osg::ApplicationUsage::COMMAND_LINE_OPTION);
    return 1;
  }

  // add the state manipulator
  viewer.addEventHandler(new osgGA::StateSetManipulator(viewer.getCamera()->getOrCreateStateSet()));

  // add the thread model handler
  viewer.addEventHandler(new osgViewer::ThreadingHandler);

  // add the window size toggle handler
  viewer.addEventHandler(new osgViewer::WindowSizeHandler);

  // add the stats handler
  viewer.addEventHandler(new osgViewer::StatsHandler);

  // add the help handler
  viewer.addEventHandler(new osgViewer::HelpHandler(arguments.getApplicationUsage()));

  // add the record camera path handler
  viewer.addEventHandler(new osgViewer::RecordCameraPathHandler);

  // add the LOD Scale handler
  viewer.addEventHandler(new osgViewer::LODScaleHandler);

  // add the screen capture handler
  viewer.addEventHandler(new osgViewer::ScreenCaptureHandler);

  viewer.getCamera()->setClearColor(osg::Vec4(0.2f, 0.2f, 0.4f, 1.f));
  viewer.setThreadingModel(osgViewer::ViewerBase::SingleThreaded);

  std::string cityName;
  arguments.read("--city", cityName);

  std::string buildings = "data/" + cityName + ".osgb";
  std::string terrain = "data/" + cityName + "_tile.osgb";

  if (buildings.empty() || terrain.empty())
  {
    std::cerr << "Error: specific command arguments expected" << std::endl;
    return 1;
  }

  osg::Group* root = createShadowScene(buildings, terrain, &viewer);

#if !HARDCODED_WINDOW_SIZE
  readDisplay(width, height);
#endif
  osg::Group* screen = createScreenCamera(root, width, height);

  // any option left unread are converted into errors to write out later.
  arguments.reportRemainingOptionsAsUnrecognized();

  // report any errors if they have occurred when parsing the program arguments.
  if (arguments.errors())
  {
    arguments.writeErrorMessages(std::cout);
    return 1;
  }

  viewer.setSceneData(screen);
  viewer.realize();
  viewer.setCameraManipulator(new osgGA::TrackballManipulator);

  while (!viewer.done())
  {
    viewer.frame();
  }

  return 0;
}
