#!/usr/bin/python
import sys, os, pdb, argparse, traceback
from warmups import *
from tests import *


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",help="Which to test")
    parser.add_argument("--allwarmups",help="Run all warmups",action="store_true",default=False)
    parser.add_argument("--alltests",help="Run all tests",action="store_true",default=False)
    parser.add_argument("--allconvs",help="Run all convolution tests",action="store_true",default=False)
    parser.add_argument("--store",help="Overwrite cache of answers",action="store_true",default=False)
    parser.add_argument("--pdb",help="Launch PDB when answers don't match",action="store_true",default=False)
    args = parser.parse_args()
    if not (args.allwarmups or args.alltests or args.allconvs):
        if args.test is None:
            print("Need a test to run")
            sys.exit(1)
    return args

def storeMultiRes(testName,resultsDict):
    #given resultsDict is a dictionary of str(seed): result
    if not os.path.exists(STORAGE):
        os.mkdir(STORAGE)
    resName = "%s/%s.npz" % (STORAGE,testName)
    np.savez(resName,**resultsDict)

def loadMultiRes(testName):
    resName = "%s/%s.npz" % (STORAGE,testName)
    if not os.path.exists(resName):
        print("Can't find result %s, giving up" % testName)
        sys.exit(1)

    results = np.load(resName)
    return {k:results[k] for k in results.keys()}


def getTests(args):
    #Return name and tests to run
    if args.alltests: #running all tests
        return "all tests", ["p%d" % i for i in range(1,21)]
    elif args.allwarmups: #running all warmups
        return "warmup tests", ["b%d" % i for i in range(1,21)]
    elif args.allconvs: #running all conv tests
        if not CONV_TESTS:
            print("If you want the convolution tests -- set CONV_TESTS in common.py to true")
            sys.exit(1)
        return "convolution tests", ["c%d" % i for i in range(1,5)]
    return "some tests",args.test.split(",")

if __name__ == "__main__":
    STORAGE = "442solutions/"
    SEEDS = [442,1337,31415,3777,2600]
    SEEDS += [s * 2 for s in SEEDS] + [s * 5 for s in SEEDS]

    args = parseArgs()

    testName, toRun = getTests(args)

    successes = 0
    for fnName in toRun:
        print("Running %s" % fnName)

        if ("gen_"+fnName) not in locals() or fnName not in locals():
            print("Function %s doesn't exist!" % fnName)
            continue

        success = True
        resultsDict = {}
        for seed in SEEDS:
            np.random.seed(seed)
            data = locals()["gen_"+fnName]()

            try:
                res = locals()[fnName](data)
            except Exception as exc:
                print("Crashed! On %s seed %d" % (fnName,seed))
                print(traceback.format_exc())
                print(exc)
                success = False
                break

            if res is None:
                print("Not implemented %s or returned None on seed %d" % (fnName,seed))
                success = False
                break

            resultsDict[str(seed)] = res

        #don't bother if it didn't work
        if not success:
            continue
        
        if args.store:
            storeMultiRes(fnName,resultsDict)


        solnDict = loadMultiRes(fnName)
        success = True
   
        for seed in SEEDS:
            strSeed = str(seed)
            res, soln = resultsDict[strSeed], solnDict[strSeed]

            if res.shape != soln.shape:
                print("\tWrong shape! On %s seed %d" % (fnName,seed))
                print("\tGiven: %s" % str(res.shape))
                print("\tExpected: %s" % str(soln.shape))
                if args.pdb:
                    print("\tCredited response: soln\n\tYour response: res")
                    pdb.set_trace()
                success = False
                break

            if fnName == "p2":
                #force signs to match for eigenvector problem
                res *= np.sign(res[0])*np.sign(soln[0])

            if not np.allclose(res,soln,rtol=1e-3,atol=1e-4):
                print("\tWrong values! On %s seed %d" % (fnName,seed))
                if args.pdb:
                    print("\tCredited response: soln\n\tYour response: res")
                    pdb.set_trace()
                success = False
                break

            if res.dtype.kind != soln.dtype.kind:
                print("\tNot the same kind! On %s seed %d" % (fnName,seed))
                print("\tGiven: %s" % res.dtype.name)
                print("\tExpected: %s" % soln.dtype.name)

        successes += 1 if success else 0

    print("Ran %s" % testName)
    print("%d/%d = %.1f" % (successes,len(toRun),100.0*successes/len(toRun)))


