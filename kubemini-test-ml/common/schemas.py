"""Pydantic schemas shared across churn services."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    """Features used by the churn model pipeline."""

    age: Optional[float] = Field(None, ge=0)
    businesstravel: Optional[str] = Field(None, description="Travel frequency category")
    dailyrate: Optional[float] = Field(None, ge=0)
    department: Optional[str] = None
    distancefromhome: Optional[float] = Field(None, ge=0)
    education: Optional[float] = Field(None, ge=0)
    educationfield: Optional[str] = None
    environmentsatisfaction: Optional[float] = Field(None, ge=0)
    gender: Optional[str] = None
    hourlyrate: Optional[float] = Field(None, ge=0)
    jobinvolvement: Optional[float] = Field(None, ge=0)
    joblevel: Optional[float] = Field(None, ge=0)
    jobrole: Optional[str] = None
    disobediencerules: Optional[str] = Field(None, description="Whether employee disobeys rules")
    jobsatisfaction: Optional[float] = Field(None, ge=0)
    maritalstatus: Optional[str] = None
    monthlyincome: Optional[float] = Field(None, ge=0)
    monthlyrate: Optional[float] = Field(None, ge=0)
    numcompaniesworked: Optional[float] = Field(None, ge=0)
    overtime: Optional[str] = Field(None, description="Yes or No")
    percentsalaryhike: Optional[float] = Field(None, ge=0)
    performancerating: Optional[float] = Field(None, ge=0)
    relationshipsatisfaction: Optional[float] = Field(None, ge=0)
    stockoptionlevel: Optional[float] = Field(None, ge=0)
    totalworkingyears: Optional[float] = Field(None, ge=0)
    trainingtimeslastyear: Optional[float] = Field(None, ge=0)
    worklifebalance: Optional[float] = Field(None, ge=0)
    yearsatcompany: Optional[float] = Field(None, ge=0)
    yearsincurrentrole: Optional[float] = Field(None, ge=0)
    yearssincelastpromotion: Optional[float] = Field(None, ge=0)
    yearswithcurrmanager: Optional[float] = Field(None, ge=0)

    class Config:
        extra = "allow"
        json_schema_extra = {
            "example": {
                "age": 34,
                "businesstravel": "Travel_Rarely",
                "dailyrate": 1020,
                "department": "Research & Development",
                "distancefromhome": 10,
                "education": 3,
                "educationfield": "Medical",
                "environmentsatisfaction": 4,
                "gender": "Male",
                "hourlyrate": 60,
                "jobinvolvement": 3,
                "joblevel": 2,
                "jobrole": "Laboratory Technician",
                "disobediencerules": "No",
                "jobsatisfaction": 3,
                "maritalstatus": "Married",
                "monthlyincome": 4500,
                "monthlyrate": 14000,
                "numcompaniesworked": 2,
                "overtime": "No",
                "percentsalaryhike": 12,
                "performancerating": 3,
                "relationshipsatisfaction": 3,
                "stockoptionlevel": 1,
                "totalworkingyears": 8,
                "trainingtimeslastyear": 3,
                "worklifebalance": 3,
                "yearsatcompany": 6,
                "yearsincurrentrole": 4,
                "yearssincelastpromotion": 1,
                "yearswithcurrmanager": 3,
            }
        }
