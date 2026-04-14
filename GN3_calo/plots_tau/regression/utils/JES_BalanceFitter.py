#!/usr/bin/env python

import ROOT
#import math

class JES_BalanceFitter():

  ## Initialize the class, with Gaussian fits by default ##
  def __init__(self, Nsigma):
    self.param="gaus"
    self.fitDesc="Gaussian"
    self.fitOpt = "RQ0"
    self.Nsigma = Nsigma
    self.minPt = 0
    self.fit = 0
    self.histo = 0
    self.fitHist = 0
    self.fitCol = ROOT.kRed
    self.lNlines = 0
    self.rNlines = 0
    self.fitMin = 0.2
    self.fitMax = 2.0

    self.PrintText = False
    self.textSize = 0.04
    self.xLeft = 0.15
    self.xRight = 0.7
    self.yTop = 0.9
    self.dy = 0.05


    self.fitMeanMin = 0.
    self.fitMeanMax = 2.
    self.fitWidthMin = 0.
    self.fitWidthMax = 1.

    self.rebin = True
    self.debug = False
    self.smartFit = False
    self.useRangeFromShape = False
    self.fractionMaxBin = 0.25
    self.fractionMaxBinLeft = None
    self.fractionMaxBinRight = None

######## Main Fit functions ##########

  ## The callable fit function.  Use this in your analysis code ##
  def Fit(self, histo, fitMin = None, fitMax = None):
    self.histo = histo
    self.fitHist = histo.Clone()

    ## Determine and perform the optimal rebinning ##
    if( self.rebin ):
      self.OptimalRebin(self.fitHist)

    f = self.BasicFit( self.fitHist, fitMin, fitMax )

    return f

  ## The basic fit ##
  def BasicFit( self, histo, fitMin = None, fitMax = None ):

    if( fitMin == None ):
      fitMin = self.fitMin
    if( fitMax == None ):
      fitMax = self.fitMax

    if( self.debug ):
      print("Running BasicFit with", histo.GetName(), fitMin, fitMax)

    ## Create and configure fit object ##
    self.fit = ROOT.TF1( "fit", self.param, fitMin, fitMax )
    self.fit.SetLineWidth(3)
    self.fit.SetParameters( histo.GetMaximum(), histo.GetMean(), histo.GetRMS() )
    self.fit.SetParLimits(1, self.fitMeanMin, self.fitMeanMax)
    self.fit.SetParLimits(2, self.fitWidthMin, self.fitWidthMax)
    self.fit.SetLineColor(self.fitCol)

    ## A smarter fit that starts with a likelihood fit and then performs two refits, each
    ## adjusting the fit ranges
    if( self.smartFit ):
      histo.Fit(self.fit,"RLQ0")

      self.SetSmartFitRange(fitMin, fitMax)
      histo.Fit(self.fit,self.fitOpt)

      self.SetSmartFitRange(fitMin, fitMax)
      histo.Fit(self.fit,self.fitOpt)
      
    else:
      if self.useRangeFromShape:
        self.SetFitRangeFromShape(histo)

      histo.Fit(self.fit,self.fitOpt)

    return self.fit

  def SetFitRangeFromShape(self, hist):
    maxContent = hist.GetBinContent(hist.GetMaximumBin())
    
    fractionLeft = self.fractionMaxBinLeft if self.fractionMaxBinLeft else self.fractionMaxBin
    firstBinAbove = hist.FindFirstBinAbove(maxContent*fractionLeft)    
    fitLow = hist.GetXaxis().GetBinLowEdge( firstBinAbove-1 )

    fractionRight = self.fractionMaxBinRight if self.fractionMaxBinRight else self.fractionMaxBin
    lastBinAbove = hist.FindLastBinAbove(maxContent*fractionRight)
    fitHigh = hist.GetXaxis().GetBinUpEdge( lastBinAbove+1 )
    
    self.fit.SetRange(fitLow, fitHigh)
  
  ## Set a smarter fit range after an initial fit ##
  def SetSmartFitRange( self, smallestMin = None, largestMax = None ):
    mean = self.GetMean()
    sigma = self.GetSigma()
    smartMin = mean - self.Nsigma*sigma
    smartMax = mean + self.Nsigma*sigma

    # Only use smart range if it is narrower than preconfigured fit range
    if ( smallestMin and smartMin < smallestMin):
      smartMin = smallestMin
    if ( largestMax and smartMax > largestMax):
      smartMax = largestMax

    self.fit.SetRange(smartMin,smartMax)
    if(self.debug):
      print("Try to find smart fit range from information of basic fit:") 
      print("mean: {:f}, sigma: {:f}".format(mean, sigma))
      print("Setting smart fit edges to", smartMin, "->", smartMax)

    return

#//-----------------------------------------------------------------------//
#// Functions to retrieve fit variables
#//-----------------------------------------------------------------------//
  def GetFit(self):
    if (self.fit==None):
      print("Something went wrong. Can't access fit function!") 
    return self.fit

  def GetHisto(self):
    if (self.fitHist==None):
      print("Something went wrong. Can't access fitted histogram!") 
    return self.fitHist

  def GetFineHisto(self):
    if (self.histo==None):
      print("Something went wrong. Can't access fitted histogram!") 
    return self.histo

  def GetMean(self):
    mean = self.GetFit().GetParameter(1)
    return mean

  def GetMeanError(self):
    param = self.GetFit().GetParError(1)
    return param

  def GetSigma(self):
    param = self.GetFit().GetParameter(2)
    return param

  def GetSigmaError(self):
    param = self.GetFit().GetParError(2)
    return param

  def GetPeak(self):
    return self.GetFit().GetMaximumX(0,2)

  def GetMedian(self):
    return self.GetQuantile(0.5)

  def getQuantile( self, frac ):
    if( frac <=0 or frac >= 1):
      print("Error, can't access quantile for fraction ", frac) 
      exit(1)

    thisMin = self.GetMean() - 5.*self.GetSigma()
    thisMax = self.GetMean() + 5.*self.GetSigma()

    if( "Pois" in self.param and thisMin < 0):
      thisMin = 0

    Nsteps = 10000

    fit = self.GetFit()
    Atot = fit.Integral(thisMin, thisMax)
    A = 0
    dx = (thisMax-thisMin)/Nsteps

    for i in range(0, Nsteps):
      xi = thisMin + i*dx
      A += fit.Eval(xi)*dx

      if( A > Atot * frac ):
        return xi - 0.5*dx

    print("Error, GetQuantile failed") 
    return 0

  def GetHistoQuantile( frac ):
    if( frac <=0 or frac >=1 ):
      print("Error, can't access quantile for fraction ", frac) 
      exit(1)

    h = self.GetFineHisto()
    h2 = h.Clone()

    #include underflow and overflow bin contents in the computation of the quantiles ("GetQuantiles" function in root ignores underflow and overflow bin contents)
    Nbins = h2.GetNbinsX()
    h2.SetBinContent( 1, h2.GetBinContent(0)+h2.GetBinContent(1) )
    h2.SetBinContent( Nbins, h2.GetBinContent(Nbins)+h2.GetBinContent(Nbins+1) )
    h2.ComputeIntegral()

    integral = h2.GetSumOfWeights()
    if( self.debug ):
      print("Computing quantiles...") 
      print("integral =", integral, "entries =", h2.GetEntries(), "underflow =", h2.GetBinContent(0), "overflow =", h2.GetBinContent(Nbins+1)) 


    nq = 1
    # position where to compute the quantiles in [0,1]
    xq = array.array('d',[0 for i in range(0, nq)])
    # array to contain the quantiles
    yq = array.array('d',[0 for i in range(0, nq)])
    xq[0] = frac


    if( h2.GetBinContent(0)/integral > xq[0] ):
      print("Error, quantile will be biassed (underflow). Consider extending the range of the histogram towards lower values.") 
      exit(1)
    if( h2.GetBinContent(Nbins+1) / integral > (1. - xq[0]) ):
      print("Error, quantile will be biassed (overflow). Consider extending the range of the histogram towards larger values.") 
      exit(1)

    h2.GetQuantiles(nq, yq, xq)
    if(self.debug):
      print("Requested quantile: " , xq[0] , "  Result: " , yq[0]) 

    return yq[0]

#//-----------------------------------------------------------------------//
  def GetChi2(self):
    return self.GetFit().GetChisquare()
#//-----------------------------------------------------------------------//
  def GetNdof(self):
    return self.GetFit().GetNDF()
#//-----------------------------------------------------------------------//
  def GetChi2Ndof(self):
    if( self.GetNdof() == 0):
      return float('inf')
      #return math.inf
    else:
      return self.GetChi2()/self.GetNdof()

  def GetChi2Prob( self ):
    return ROOT.TMath.Prob( self.GetChi2(), self.GetNdof() )

#//-----------------------------------------------------------------------//
#// Functions to draw and print fit info
#//---------------------------------------------------------------------//
  def DrawFitAndHisto(self, drawRangeMin = None, drawRangeMax = None):
    xlow = self.GetMean()-7*self.GetSigma() if not drawRangeMin else drawRangeMin
    xup = self.GetMean()+7*self.GetSigma() if not drawRangeMax else drawRangeMax
    self.GetHisto().GetXaxis().SetRangeUser(xlow, xup)
    self.GetHisto().Draw()
    self.DrawExtendedFit()
    self.ResetTextCounters()

    if( self.PrintText ):
      self.PrintFitInfo()

  def ResetTextCounters(self):
    self.lNlines=0
    self.rNlines=0

  ## Print fit information to the canvas ##
  def PrintFitInfo(self):
    self.DrawTextRight(self.fitDesc,self.fitCol)
    self.DrawTextRight( 'chi/dof: {:.2f}'.format(self.GetChi2Ndof()), self.fitCol)
    self.DrawTextRight( "mean: ({:.2f}#pm{:.2f})%".format( (self.GetMean()-1.0)*100, self.GetMeanError()*100), self.fitCol)
    self.DrawTextRight( "width: ({:.2f}#pm{:.2f})%".format( self.GetSigma()*100, self.GetSigmaError()*100), self.fitCol)

  ## Draw the fit extended past the fit range ##
  def DrawExtendedFit(self, minx = 0, maxx = 3):
    fit = self.GetFit()

    lowX, highX = ROOT.Double(0), ROOT.Double(0)
    fit.GetRange( lowX, highX )
    lineStyle = fit.GetLineStyle()

    fit.SetRange(minx,maxx)
    fit.SetLineStyle(2)
    fit.DrawCopy("same")

    fit.SetRange(lowX, highX)
    fit.SetLineStyle(lineStyle)

    self.GetFit().Draw("same")
    return

  def DrawTextLeft(self, txt, col):
    self.DrawText(self.xLeft,self.yTop-self.dy*(self.lNlines),txt,col)
    self.lNlines += 1
    return

  def DrawTextRight(self, txt, col):
    self.DrawText(self.xRight,self.yTop-self.dy*(self.rNlines),txt,col)
    self.rNlines += 1
    return

  def DrawText(self, x, y, txt, col):
    tex = ROOT.TLatex()
    tex.SetNDC()
    tex.SetTextFont(42)
    tex.SetTextSize(self.textSize)
    tex.SetTextColor(col)
    tex.DrawLatex(x,y,txt)
    return

  def OptimalRebin( self, h ):
    method = 1
    N = h.GetEffectiveEntries()
    if N == 0:
      print("WARNING: Cannot do optimal rebinning due to effective entries being zero!") 
      return
    optWidth = 3.5*h.GetRMS()/ROOT.TMath.Power(N,1.0/3)
    Nbins=int(h.GetNbinsX())
    fitRange = h.GetBinLowEdge(Nbins+1) - h.GetBinLowEdge(1)
    rebin=1
    prevWidth=fitRange/Nbins
    for i in range(1, Nbins) :
      if (Nbins%i!=0):
        continue
      binWidth=fitRange/Nbins*i

      if (method==1):
        if (binWidth<optWidth):
           rebin=i
      elif (method==2):
        if (ROOT.TMath.Abs(binWidth-optWidth) < ROOT.TMath.Abs(prevWidth-optWidth)):
          rebin=i
      else:
        rebin=i #// method 3

      if (binWidth>optWidth):
         break
      prevWidth=binWidth

    _vebose=False
    if (_vebose):
      printf("\n%s\n  RMS: %.3f, Neff: %.3f\n",h.GetName(),h.GetRMS(),N)
      printf("  Opt width: %6.3f, histo binwidth: %6.3f => Rebin: %d\n",optWidth,fitRange/Nbins,rebin)
    self.fitHist.Rebin(rebin)
    return

######  Set attributes of fitter ####
  def SetFitColor(self, col):
    self.fitCol = col
    return

  def SetGaus(self):
    self.param="gaus"
    self.fitDesc="Gaussian"

  def SetPoisson(self):
    self.param="[0]*2.8*TMath::Poisson(x/pow([2],2),[1]/pow([2],2))"
    self.fitDesc="Modified Poisson"

  def SetRebin( self, doRebin ):
    self.rebin = doRebin
    return

